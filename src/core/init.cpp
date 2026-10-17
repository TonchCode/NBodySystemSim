// Initial conditions and host-side diagnostics (double precision).
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include "nbody.h"

namespace nbody {

// Plummer model, Aarseth, Henon & Wielen (1974) sampling, scaled to N-body units
// (G = M = 1, E = -1/4). Positions in units of the virial radius.
void makePlummer(int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double kPi   = 3.14159265358979323846;
    const double scale = 3.0 * kPi / 16.0;   // Plummer radius -> virial units
    const double m     = 1.0 / n;
    pos.resize(n);
    vel.resize(n);

    auto randomDir = [&](double r, double& x, double& y, double& z) {
        double cz  = 2.0 * U(rng) - 1.0;
        double sz  = std::sqrt(1.0 - cz * cz);
        double phi = 2.0 * kPi * U(rng);
        x = r * sz * std::cos(phi);
        y = r * sz * std::sin(phi);
        z = r * cz;
    };

    for (int i = 0; i < n; ++i) {
        double r;
        do { r = 1.0 / std::sqrt(std::pow(U(rng), -2.0 / 3.0) - 1.0); } while (r > 10.0);  // cut far tail
        double x, y, z;
        randomDir(r, x, y, z);

        // rejection sampling of q = v / v_escape from g(q) = q^2 (1 - q^2)^3.5
        double q, g;
        do { q = U(rng); g = 0.1 * U(rng); } while (g > q * q * std::pow(1.0 - q * q, 3.5));
        double v = q * std::sqrt(2.0) * std::pow(1.0 + r * r, -0.25);
        double vx, vy, vz;
        randomDir(v, vx, vy, vz);

        pos[i] = make_float4(float(x * scale), float(y * scale), float(z * scale), float(m));
        double vs = 1.0 / std::sqrt(scale);
        vel[i] = make_float4(float(vx * vs), float(vy * vs), float(vz * vs), 0.f);
    }

    // remove centre-of-mass drift
    double cx = 0, cy = 0, cz = 0, cvx = 0, cvy = 0, cvz = 0;
    for (int i = 0; i < n; ++i) {
        cx += pos[i].x; cy += pos[i].y; cz += pos[i].z;
        cvx += vel[i].x; cvy += vel[i].y; cvz += vel[i].z;
    }
    cx /= n; cy /= n; cz /= n; cvx /= n; cvy /= n; cvz /= n;
    for (int i = 0; i < n; ++i) {
        pos[i].x -= float(cx); pos[i].y -= float(cy); pos[i].z -= float(cz);
        vel[i].x -= float(cvx); vel[i].y -= float(cvy); vel[i].z -= float(cvz);
    }
}

namespace {

constexpr double kPi = 3.14159265358979323846;

// One Plummer component: makePlummer (virial units, mass 1) scaled to virial radius a and
// mass mc (G = 1: velocities scale with sqrt(mc / a)), moved to centre c with bulk velocity v.
// Every body gets mass mBody (the legacy ICs pass 1/nTotal, so every body in them has the same mass).
void appendPlummer(int nc, unsigned seed, double a, double mc, double3 c, double3 v, double mBody,
                   std::vector<float4>& pos, std::vector<float4>& vel)
{
    std::vector<float4> p, u;
    makePlummer(nc, seed, p, u);
    const double vs = std::sqrt(mc / a);
    for (int i = 0; i < nc; ++i) {
        pos.push_back(make_float4(float(c.x + a * p[i].x), float(c.y + a * p[i].y), float(c.z + a * p[i].z),
                                  float(mBody)));
        vel.push_back(make_float4(float(v.x + vs * u[i].x), float(v.y + vs * u[i].y), float(v.z + vs * u[i].z), 0.f));
    }
}

// A contiguous run [begin, end) of the (pre-shuffle) arrays that forms one physical component.
// spectrum = false: the component's masses are never redrawn (the central massive body).
struct Component { int begin, end; bool spectrum; };

// Shuffle so the original order is spatially random (like the single Plummer): otherwise the
// components would be contiguous runs and bh_naive's "unsorted" warps would be partly sorted.
void shuffleBodies(int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel)
{
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937(seed ^ 0x5eedu));
    std::vector<float4> p2(n), v2(n);
    for (int i = 0; i < n; ++i) { p2[i] = pos[idx[i]]; v2[i] = vel[idx[i]]; }
    pos.swap(p2); vel.swap(v2);
}

// The legacy ICs, unshuffled, with their component runs (bitwise the pre-IC-round code path).
bool buildLegacy(const std::string& name, int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel,
                 std::vector<Component>& comps)
{
    pos.clear(); vel.clear(); comps.clear();
    pos.reserve(n); vel.reserve(n);
    if (name == "plummer") {
        makePlummer(n, seed, pos, vel);
        comps.push_back({0, n, true});
    } else if (name == "collision") {
        const int n1 = n / 2, n2 = n - n1;
        appendPlummer(n1, seed, 0.25, 0.5, make_double3(-1.0, -0.125, 0.0), make_double3(0.35, 0.0, 0.0), 1.0 / n, pos, vel);
        appendPlummer(n2, seed + 1000003u, 0.25, 0.5, make_double3(1.0, 0.125, 0.0), make_double3(-0.35, 0.0, 0.0), 1.0 / n,
                      pos, vel);
        comps.push_back({0, n1, true});
        comps.push_back({n1, n, true});
    } else if (name == "core") {
        const int nc = n / 4, nh = n - nc;
        appendPlummer(nh, seed, 1.0, 0.75, make_double3(0, 0, 0), make_double3(0, 0, 0), 1.0 / n, pos, vel);
        appendPlummer(nc, seed + 1000003u, 0.03, 0.25, make_double3(0, 0, 0), make_double3(0, 0, 0), 1.0 / n, pos, vel);
        comps.push_back({0, nh, true});
        comps.push_back({nh, n, true});
    } else {
        return false;
    }
    return true;
}

// Mass spectrum (nbody.h): redraws the masses of every spectrum component, independently of
// position, normalised so the component keeps its mass (the sum of its current masses).
void applySpectrum(const std::string& sp, const std::vector<Component>& comps, unsigned seed, std::vector<float4>& pos)
{
    if (sp == "equal") return;
    for (size_t c = 0; c < comps.size(); ++c) {
        const Component& k = comps[c];
        const int nc = k.end - k.begin;
        if (!k.spectrum || nc <= 0) continue;
        double mc = 0;
        for (int i = k.begin; i < k.end; ++i) mc += pos[i].w;
        std::vector<double> m(nc);
        if (sp == "salpeter") {   // inverse CDF of dN/dm ~ m^-2.35 on [1, 100]
            std::mt19937 rng(seed * 2654435761u + 0x9e3779b9u + unsigned(c));
            std::uniform_real_distribution<double> U(0.0, 1.0);
            const double a1 = 1.0 - 2.35, A = std::pow(1.0, a1), B = std::pow(100.0, a1);
            double s = 0;
            for (double& x : m) { x = std::pow(A + U(rng) * (B - A), 1.0 / a1); s += x; }
            for (double& x : m) x *= mc / s;
        } else {   // heavy-few: the first H bodies of the run (iid positions, so a random choice)
            const int H = std::min(16, nc / 128);
            const double mh = 0.01 * mc, ml = nc > H ? (mc - H * mh) / (nc - H) : 0.0;
            for (int i = 0; i < nc; ++i) m[i] = i < H ? mh : ml;
        }
        for (int i = 0; i < nc; ++i) pos[k.begin + i].w = float(m[i]);
    }
}

// Mass-weighted centre of mass and momentum removed (new ICs and every non-equal spectrum).
void recentre(std::vector<float4>& pos, std::vector<float4>& vel)
{
    double M = 0, cx = 0, cy = 0, cz = 0, px = 0, py = 0, pz = 0;
    for (size_t i = 0; i < pos.size(); ++i) {
        const double m = pos[i].w;
        M += m; cx += m * pos[i].x; cy += m * pos[i].y; cz += m * pos[i].z;
        px += m * vel[i].x; py += m * vel[i].y; pz += m * vel[i].z;
    }
    if (!(M > 0)) return;
    cx /= M; cy /= M; cz /= M; px /= M; py /= M; pz /= M;
    for (size_t i = 0; i < pos.size(); ++i) {
        pos[i].x = float(pos[i].x - cx); pos[i].y = float(pos[i].y - cy); pos[i].z = float(pos[i].z - cz);
        vel[i].x = float(vel[i].x - px); vel[i].y = float(vel[i].y - py); vel[i].z = float(vel[i].z - pz);
    }
}

struct Rot { double m[9]; };   // row-major 3x3
Rot rotX(double a) { const double c = std::cos(a), s = std::sin(a); return {{1, 0, 0, 0, c, -s, 0, s, c}}; }
Rot rotY(double a) { const double c = std::cos(a), s = std::sin(a); return {{c, 0, s, 0, 1, 0, -s, 0, c}}; }
Rot mul(const Rot& a, const Rot& b)
{
    Rot r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) r.m[3 * i + j] += a.m[3 * i + k] * b.m[3 * k + j];
    return r;
}
void apply(const Rot& r, double& x, double& y, double& z)
{
    const double a = r.m[0] * x + r.m[1] * y + r.m[2] * z, b = r.m[3] * x + r.m[4] * y + r.m[5] * z,
                 c = r.m[6] * x + r.m[7] * y + r.m[8] * z;
    x = a; y = b; z = c;
}

// Disk galaxy (docs/03-design.md §19.1), in its own frame: disk in the x-y plane rotating about +z,
// then rotated by R, moved to centre c with bulk velocity v. Components (in order): central body,
// bulge, disk. Scales (N-body units, G = 1): disk scale length Rd = 0.25, truncated at 6 Rd, sech^2
// thickness z0 = 0.1 Rd; Plummer bulge with virial radius 0.08 holding 10% of the light bodies;
// central body = bhMass of the galaxy's mass, at rest at the centre. Light bodies all have the same
// mass before the spectrum.
// Velocities: v_c(R)^2 = M(<r) R^2 / (r^2 + eps^2)^1.5 with M(<r) the mass enclosed in the SPHERE of
// radius r = |x| (counted from the sampled bodies), i.e. the softened circular speed of a spherical
// distribution with the same M(r). Exact for the central body and the bulge, an approximation for the
// disk's own field [TODO(source): Freeman 1970, exact thin exponential disk rotation curve; not in
// references.bib]. Dispersions: sigma_R = sigma_phi = 0.15 v_c; sigma_z^2 = pi G Sigma(R) z0
// (isothermal-sheet relation [TODO(source): Spitzer 1942; not in references.bib]). Bulge: the isolated
// Plummer velocities scaled by sqrt(M_total(<r) / M_bulge(<r)) (a heuristic, not a Jeans solution).
// No dark halo, no Toomre-Q control (Q << 1 at these dispersions): the disk is self-gravitating and
// will form bars, spirals and clumps within about one rotation; it is a force snapshot for the
// workload study and a demo for the viewer, not a stable Milky Way.
void appendDisk(int ng, unsigned seed, double Mg, double3 c, double3 v, const Rot& R, const ICOptions& o,
                std::vector<Component>& comps, std::vector<float4>& pos, std::vector<float4>& vel)
{
    if (ng <= 0) return;
    const double Rd = 0.25, Rmax = 6.0 * Rd, z0 = 0.1 * Rd, ab = 0.08, fb = 0.1;
    const double eps2 = o.softening * o.softening;
    const int    nbh  = o.bhMass > 0 ? 1 : 0;
    const int    nl   = ng - nbh;
    const double mbh  = nbh ? (nl > 0 ? o.bhMass * Mg : Mg) : 0.0;
    const double ml   = nl > 0 ? (Mg - mbh) / nl : 0.0;
    const int    nb   = int(std::lround(fb * nl)), nd = nl - nb;
    std::vector<double> X(ng), Y(ng), Z(ng), VX(ng, 0.0), VY(ng, 0.0), VZ(ng, 0.0), M(ng);
    std::mt19937 rng(seed * 747796405u + 2891336453u);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> Ng(0.0, 1.0);
    int k = 0;
    if (nbh) { X[0] = Y[0] = Z[0] = 0.0; M[0] = mbh; k = 1; }
    std::vector<float4> bp, bv;
    makePlummer(nb, seed + 17u, bp, bv);
    const int b0 = k;
    for (int i = 0; i < nb; ++i, ++k) { X[k] = ab * bp[i].x; Y[k] = ab * bp[i].y; Z[k] = ab * bp[i].z; M[k] = ml; }
    const int d0 = k;
    for (int i = 0; i < nd; ++i, ++k) {
        double r;
        do { r = -Rd * std::log(std::max(1e-300, U(rng) * U(rng))); } while (r > Rmax);   // Sigma ~ exp(-R/Rd)
        const double ph = 2.0 * kPi * U(rng);
        const double uz = std::min(std::max(U(rng), 1e-12), 1.0 - 1e-12);
        X[k] = r * std::cos(ph); Y[k] = r * std::sin(ph); Z[k] = z0 * std::atanh(2.0 * uz - 1.0);   // rho ~ sech^2(z/z0)
        M[k] = ml;
    }
    // enclosed mass in spheres, from the realised bodies (total and bulge only)
    std::vector<std::pair<double, double>> rt(ng), rb(nb);
    for (int i = 0; i < ng; ++i) rt[i] = {std::sqrt(X[i] * X[i] + Y[i] * Y[i] + Z[i] * Z[i]), M[i]};
    for (int i = 0; i < nb; ++i) rb[i] = rt[b0 + i];
    std::sort(rt.begin(), rt.end());
    std::sort(rb.begin(), rb.end());
    std::vector<double> ct(ng + 1, 0.0), cb(nb + 1, 0.0);
    for (int i = 0; i < ng; ++i) ct[i + 1] = ct[i] + rt[i].second;
    for (int i = 0; i < nb; ++i) cb[i + 1] = cb[i] + rb[i].second;
    auto menc = [](const std::vector<std::pair<double, double>>& s, const std::vector<double>& cum, double r) {
        const size_t j = size_t(std::lower_bound(s.begin(), s.end(), std::make_pair(r, -1.0)) - s.begin());
        return cum[j];   // bodies strictly inside r (a body does not pull itself)
    };
    const double Md = nd * ml;
    for (int i = d0; i < ng; ++i) {
        const double Rc = std::sqrt(X[i] * X[i] + Y[i] * Y[i]), r = std::sqrt(Rc * Rc + Z[i] * Z[i]);
        const double vc = std::sqrt(std::max(0.0, menc(rt, ct, r) * Rc * Rc / std::pow(r * r + eps2, 1.5)));
        const double sz = std::sqrt(kPi * Md / (2.0 * kPi * Rd * Rd) * std::exp(-Rc / Rd) * z0);
        const double cp = Rc > 0 ? X[i] / Rc : 1.0, sp = Rc > 0 ? Y[i] / Rc : 0.0;
        const double vr = 0.15 * vc * Ng(rng), vp = vc + 0.15 * vc * Ng(rng);
        VX[i] = vr * cp - vp * sp; VY[i] = vr * sp + vp * cp; VZ[i] = sz * Ng(rng);
    }
    const double vsb = nb > 0 ? std::sqrt((nb * ml) / ab) : 0.0;
    for (int i = 0; i < nb; ++i) {
        const int j = b0 + i;
        const double r = std::sqrt(X[j] * X[j] + Y[j] * Y[j] + Z[j] * Z[j]);
        const double mb = menc(rb, cb, r), mt = menc(rt, ct, r);
        const double f  = mb > 0 ? std::sqrt(mt / mb) : 1.0;
        VX[j] = vsb * f * bv[i].x; VY[j] = vsb * f * bv[i].y; VZ[j] = vsb * f * bv[i].z;
    }
    const int base = int(pos.size());
    for (int i = 0; i < ng; ++i) {
        double x = X[i], y = Y[i], z = Z[i], vx = VX[i], vy = VY[i], vz = VZ[i];
        apply(R, x, y, z);
        apply(R, vx, vy, vz);
        pos.push_back(make_float4(float(c.x + x), float(c.y + y), float(c.z + z), float(M[i])));
        vel.push_back(make_float4(float(v.x + vx), float(v.y + vy), float(v.z + vz), 0.f));
    }
    if (nbh) comps.push_back({base, base + 1, false});
    comps.push_back({base + b0, base + d0, true});
    comps.push_back({base + d0, base + ng, true});
}

// Cube ICs: uniform on the faces / edges / volume of [-1,1]^3, small isotropic random velocities
// (sigma 0.05 per component: almost cold, so they collapse in the viewer).
void appendCube(const std::string& name, int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel)
{
    std::mt19937 rng(seed * 2246822519u + 3266489917u);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> Ng(0.0, 0.05);
    for (int i = 0; i < n; ++i) {
        double x[3];
        if (name == "uniform") {
            for (double& t : x) t = 2.0 * U(rng) - 1.0;
        } else if (name == "cube_shell") {   // face f: axis f / 2 fixed at -1 or +1
            const int f  = std::min(5, int(6.0 * U(rng)));
            const int ax = f / 2;
            x[ax] = (f & 1) ? 1.0 : -1.0;
            x[(ax + 1) % 3] = 2.0 * U(rng) - 1.0;
            x[(ax + 2) % 3] = 2.0 * U(rng) - 1.0;
        } else {   // cube_edges: axis e / 4 varies, the other two are -1/+1 from the bits of e % 4
            const int e  = std::min(11, int(12.0 * U(rng)));
            const int ax = e / 4;
            x[ax] = 2.0 * U(rng) - 1.0;
            x[(ax + 1) % 3] = (e & 1) ? 1.0 : -1.0;
            x[(ax + 2) % 3] = (e & 2) ? 1.0 : -1.0;
        }
        pos.push_back(make_float4(float(x[0]), float(x[1]), float(x[2]), float(1.0 / n)));
        const double vx = Ng(rng), vy = Ng(rng), vz = Ng(rng);
        vel.push_back(make_float4(float(vx), float(vy), float(vz), 0.f));
    }
}

} // namespace

const char* icNames() { return "plummer|collision|core|disk|galaxies|cube_shell|cube_edges|uniform"; }
const char* massSpectrumNames() { return "equal|salpeter|heavy-few"; }
bool validMassSpectrum(const std::string& s) { return s == "equal" || s == "salpeter" || s == "heavy-few"; }

bool makeIC(const std::string& name, int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel)
{
    std::vector<Component> comps;
    if (!buildLegacy(name, n, seed, pos, vel, comps)) return false;
    if (name != "plummer") shuffleBodies(n, seed, pos, vel);
    return true;
}

bool makeIC(const std::string& name, int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel,
            const ICOptions& opt)
{
    if (!validMassSpectrum(opt.massSpectrum) || !(opt.bhMass >= 0.0) || !(opt.bhMass < 1.0) || !(opt.softening > 0.0))
        return false;
    std::vector<Component> comps;
    const bool legacy = buildLegacy(name, n, seed, pos, vel, comps);
    if (legacy) {
        if (opt.massSpectrum == "equal") {   // bitwise the legacy IC
            if (name != "plummer") shuffleBodies(n, seed, pos, vel);
            return true;
        }
    } else if (name == "disk") {
        appendDisk(n, seed, 1.0, make_double3(0, 0, 0), make_double3(0, 0, 0), rotX(0.0), opt, comps, pos, vel);
    } else if (name == "galaxies") {
        // Two disks of mass 1/2, centres (-/+1.5, -/+0.5, 0), bulk velocities +/-(0.3, 0.05, 0): separation
        // 3.16, relative speed 0.60 < the mutual escape speed 0.80, so bound: a close passage, then a merger.
        // Disk 1 tilted 30 deg about x; disk 2 tilted -50 deg about x, then 40 deg about y.
        const int n1 = n / 2, n2 = n - n1;
        const double M1 = n1 > 0 ? 0.5 : 0.0, M2 = 1.0 - M1;   // n = 1: one galaxy of mass 1
        appendDisk(n1, seed, M1, make_double3(-1.5, -0.5, 0.0), make_double3(0.3, 0.05, 0.0), rotX(0.5236), opt, comps,
                   pos, vel);
        appendDisk(n2, seed + 1000003u, M2, make_double3(1.5, 0.5, 0.0), make_double3(-0.3, -0.05, 0.0),
                   mul(rotY(0.6981), rotX(-0.8727)), opt, comps, pos, vel);
    } else if (name == "cube_shell" || name == "cube_edges" || name == "uniform") {
        appendCube(name, n, seed, pos, vel);
        comps.push_back({0, n, true});
    } else {
        return false;
    }
    applySpectrum(opt.massSpectrum, comps, seed, pos);
    recentre(pos, vel);
    if (name != "plummer") shuffleBodies(n, seed, pos, vel);
    return true;
}

unsigned long long icHash(const std::vector<float4>& pos, const std::vector<float4>& vel)
{
    unsigned long long h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t bytes) {
        const unsigned char* c = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < bytes; ++i) { h ^= c[i]; h *= 1099511628211ull; }
    };
    mix(pos.data(), pos.size() * sizeof(float4));
    mix(vel.data(), vel.size() * sizeof(float4));
    return h;
}

double dtAdvisory(const std::vector<float4>& pos, float softening, float G)
{
    double mmax = 0;
    for (const float4& p : pos) mmax = std::max(mmax, double(p.w));
    const double e = softening, coef = 2.0 / std::pow(3.0, 1.5);
    return mmax > 0 ? 0.1 * std::sqrt(e * e * e / (coef * G * mmax)) : 0.0;
}

Energy computeEnergy(const std::vector<float4>& pos, const std::vector<float4>& vel, const Params& p)
{
    const int    n    = int(pos.size());
    const double eps2 = double(p.softening) * p.softening;
    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < n; ++i) {
        ke += 0.5 * pos[i].w * (double(vel[i].x) * vel[i].x + double(vel[i].y) * vel[i].y +
                                double(vel[i].z) * vel[i].z);
        for (int j = i + 1; j < n; ++j) {
            double dx = double(pos[j].x) - pos[i].x;
            double dy = double(pos[j].y) - pos[i].y;
            double dz = double(pos[j].z) - pos[i].z;
            pe -= p.G * double(pos[i].w) * pos[j].w / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
    }
    return {ke, pe, ke + pe};
}

ErrorStats compareAcc(const std::vector<float4>& acc, const std::vector<float4>& ref)
{
    double maxRel = 0.0, sumSq = 0.0;
    for (size_t i = 0; i < acc.size(); ++i) {
        double dx = double(acc[i].x) - ref[i].x;
        double dy = double(acc[i].y) - ref[i].y;
        double dz = double(acc[i].z) - ref[i].z;
        double rn = std::sqrt(double(ref[i].x) * ref[i].x + double(ref[i].y) * ref[i].y +
                              double(ref[i].z) * ref[i].z);
        double rel = std::sqrt(dx * dx + dy * dy + dz * dz) / (rn > 0 ? rn : 1.0);
        maxRel = rel > maxRel ? rel : maxRel;
        sumSq += rel * rel;
    }
    return {maxRel, std::sqrt(sumSq / double(acc.size()))};
}

Fp64Sample referenceFp64(const std::vector<float4>& pos, const Params& p, const std::vector<int>& idx)
{
    const int    n    = int(pos.size());
    const double eps2 = double(p.softening) * p.softening;
    Fp64Sample r;
    r.idx = idx;
    r.ax.resize(idx.size()); r.ay.resize(idx.size()); r.az.resize(idx.size()); r.scale.resize(idx.size());
    for (size_t k = 0; k < idx.size(); ++k) {
        const float4 bi = pos[idx[k]];
        double ax = 0, ay = 0, az = 0, sc = 0;
        for (int j = 0; j < n; ++j) {
            double dx = double(pos[j].x) - bi.x;
            double dy = double(pos[j].y) - bi.y;
            double dz = double(pos[j].z) - bi.z;
            double d2 = dx * dx + dy * dy + dz * dz + eps2;
            double s  = pos[j].w / (d2 * std::sqrt(d2));
            ax += dx * s; ay += dy * s; az += dz * s;
            sc += std::sqrt(dx * dx + dy * dy + dz * dz) * s;   // |f_ij| / m_i
        }
        r.ax[k] = p.G * ax; r.ay[k] = p.G * ay; r.az[k] = p.G * az; r.scale[k] = p.G * sc;
    }
    return r;
}

double kappaOf(const Fp64Sample& r, size_t k)
{
    double rn = std::sqrt(r.ax[k] * r.ax[k] + r.ay[k] * r.ay[k] + r.az[k] * r.az[k]);
    return rn > 0 ? r.scale[k] / rn : 0.0;
}

Fp64Error compareFp64(const std::vector<float4>& acc, const Fp64Sample& ref)
{
    Fp64Error e{0, 0, 0, 0};
    double sumSq = 0;
    for (size_t k = 0; k < ref.idx.size(); ++k) {
        const float4 a  = acc[ref.idx[k]];
        double dx = a.x - ref.ax[k], dy = a.y - ref.ay[k], dz = a.z - ref.az[k];
        double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
        double rn = std::sqrt(ref.ax[k] * ref.ax[k] + ref.ay[k] * ref.ay[k] + ref.az[k] * ref.az[k]);
        double rel = d / (rn > 0 ? rn : 1.0);
        e.maxRel   = rel > e.maxRel ? rel : e.maxRel;
        sumSq     += rel * rel;
        double nm  = d / (ref.scale[k] > 0 ? ref.scale[k] : 1.0);
        e.normMax  = nm > e.normMax ? nm : e.normMax;
        double kap = rn > 0 ? ref.scale[k] / rn : 0.0;
        e.kappaMax = kap > e.kappaMax ? kap : e.kappaMax;
    }
    e.rmsRel = ref.idx.empty() ? 0.0 : std::sqrt(sumSq / double(ref.idx.size()));
    return e;
}

} // namespace nbody
