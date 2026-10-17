// Initial conditions and host-side diagnostics (double precision).
#include <cmath>
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
