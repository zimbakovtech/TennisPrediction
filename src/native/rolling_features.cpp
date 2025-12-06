#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <deque>
#include <cmath>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace pybind11;

struct Occurrence {
    int idx;
    bool is_player_role;
    double v_ace;
    double v_df;
    double v_bp;
};

struct DequeStatEntry {
    int idx;
    double value;
};

static inline bool is_nan(double x) { return isnan(x); }

static inline double round_half_even(double x, int n) {
    if (is_nan(x)) return x;
    double f = pow(10.0, n);
    double y = x * f;
    double yf = floor(y);
    double frac = y - yf;
    if (fabs(frac - 0.5) <= 1e-12 * max(1.0, fabs(y))) {
        if (fmod(yf, 2.0) != 0.0) {
            return (yf + 1.0) / f;
        } else {
            return yf / f;
        }
    }
    return round(y) / f;
}

dict compute_rolling_features(
    array_t<long long, array::c_style | array::forcecast> player_ids,
    array_t<long long, array::c_style | array::forcecast> opponent_ids,
    array_t<double, array::c_style | array::forcecast> w_ace,
    array_t<double, array::c_style | array::forcecast> l_ace,
    array_t<double, array::c_style | array::forcecast> w_df,
    array_t<double, array::c_style | array::forcecast> l_df,
    array_t<double, array::c_style | array::forcecast> w_bpSaved,
    array_t<double, array::c_style | array::forcecast> l_bpSaved,
    int window,
    int lookback,
    int num_threads
) {
    auto t0 = chrono::high_resolution_clock::now();

    if (window <= 0) throw invalid_argument("window must be > 0");
    if (lookback <= 0) throw invalid_argument("lookback must be > 0");

    const auto n = static_cast<int>(player_ids.size());
    if (opponent_ids.size() != n || w_ace.size() != n || l_ace.size() != n ||
        w_df.size() != n || l_df.size() != n || w_bpSaved.size() != n || l_bpSaved.size() != n) {
        throw invalid_argument("All input arrays must have the same length");
    }

    unordered_map<long long, int> id_to_idx;
    id_to_idx.reserve(n * 2 / 3);
    vector<long long> idx_to_id; idx_to_id.reserve(n / 2);

    auto p_ptr = player_ids.data();
    auto o_ptr = opponent_ids.data();

    for (int i = 0; i < n; ++i) {
        long long a = p_ptr[i];
        long long b = o_ptr[i];
        if (!id_to_idx.count(a)) { id_to_idx[a] = static_cast<int>(idx_to_id.size()); idx_to_id.push_back(a); }
        if (!id_to_idx.count(b)) { id_to_idx[b] = static_cast<int>(idx_to_id.size()); idx_to_id.push_back(b); }
    }

    const int P = static_cast<int>(idx_to_id.size());

    vector<vector<Occurrence>> groups(P);
    for (int i = 0; i < n; ++i) {
        int gi_p = id_to_idx[p_ptr[i]];
        int gi_o = id_to_idx[o_ptr[i]];
        groups[gi_p].push_back({i, true,  w_ace.at(i), w_df.at(i), w_bpSaved.at(i)});
        groups[gi_o].push_back({i, false, l_ace.at(i), l_df.at(i), l_bpSaved.at(i)});
    }

    array_t<double> w_ace_avg(n); auto w_ace_avg_ptr = (double*)w_ace_avg.request().ptr;
    array_t<double> l_ace_avg(n); auto l_ace_avg_ptr = (double*)l_ace_avg.request().ptr;
    array_t<double> w_df_avg(n);  auto w_df_avg_ptr  = (double*)w_df_avg.request().ptr;
    array_t<double> l_df_avg(n);  auto l_df_avg_ptr  = (double*)l_df_avg.request().ptr;
    array_t<double> w_bp_avg(n);  auto w_bp_avg_ptr  = (double*)w_bp_avg.request().ptr;
    array_t<double> l_bp_avg(n);  auto l_bp_avg_ptr  = (double*)l_bp_avg.request().ptr;

    for (int i = 0; i < n; ++i) {
        w_ace_avg_ptr[i] = numeric_limits<double>::quiet_NaN();
        l_ace_avg_ptr[i] = numeric_limits<double>::quiet_NaN();
        w_df_avg_ptr[i]  = numeric_limits<double>::quiet_NaN();
        l_df_avg_ptr[i]  = numeric_limits<double>::quiet_NaN();
        w_bp_avg_ptr[i]  = numeric_limits<double>::quiet_NaN();
        l_bp_avg_ptr[i]  = numeric_limits<double>::quiet_NaN();
    }

    #ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int g = 0; g < P; ++g) {
        const auto &occs = groups[g];
        deque<DequeStatEntry> dq_ace;
        deque<DequeStatEntry> dq_df;
        deque<DequeStatEntry> dq_bp;
        double sum_ace = 0.0, sum_df = 0.0, sum_bp = 0.0;

        for (const auto &occ : occs) {
            int idx = occ.idx;
            int cutoff = idx - lookback;
            while (!dq_ace.empty() && dq_ace.front().idx < cutoff) { sum_ace -= dq_ace.front().value; dq_ace.pop_front(); }
            while (!dq_df.empty()  && dq_df.front().idx  < cutoff) { sum_df  -= dq_df.front().value;  dq_df.pop_front(); }
            while (!dq_bp.empty()  && dq_bp.front().idx  < cutoff) { sum_bp  -= dq_bp.front().value;  dq_bp.pop_front(); }

            double m_ace = dq_ace.empty() ? numeric_limits<double>::quiet_NaN() : (sum_ace / (double)dq_ace.size());
            double m_df  = dq_df.empty()  ? numeric_limits<double>::quiet_NaN() : (sum_df  / (double)dq_df.size());
            double m_bp  = dq_bp.empty()  ? numeric_limits<double>::quiet_NaN() : (sum_bp  / (double)dq_bp.size());

            m_ace = round_half_even(m_ace, 2);
            m_df  = round_half_even(m_df, 2);
            m_bp  = round_half_even(m_bp, 2);

            if (occ.is_player_role) {
                w_ace_avg_ptr[idx] = m_ace;
                w_df_avg_ptr[idx]  = m_df;
                w_bp_avg_ptr[idx]  = m_bp;
            } else {
                l_ace_avg_ptr[idx] = m_ace;
                l_df_avg_ptr[idx]  = m_df;
                l_bp_avg_ptr[idx]  = m_bp;
            }

            if (!is_nan(occ.v_ace)) {
                dq_ace.push_back({idx, occ.v_ace}); sum_ace += occ.v_ace; if ((int)dq_ace.size() > window) { sum_ace -= dq_ace.front().value; dq_ace.pop_front(); }
            }
            if (!is_nan(occ.v_df)) {
                dq_df.push_back({idx, occ.v_df}); sum_df += occ.v_df; if ((int)dq_df.size() > window) { sum_df -= dq_df.front().value; dq_df.pop_front(); }
            }
            if (!is_nan(occ.v_bp)) {
                dq_bp.push_back({idx, occ.v_bp}); sum_bp += occ.v_bp; if ((int)dq_bp.size() > window) { sum_bp -= dq_bp.front().value; dq_bp.pop_front(); }
            }
        }
    }

    array_t<double> ace_diff(n); auto ace_diff_ptr = (double*)ace_diff.request().ptr;
    array_t<double> df_diff(n);  auto df_diff_ptr  = (double*)df_diff.request().ptr;
    array_t<double> bp_diff(n);  auto bp_diff_ptr  = (double*)bp_diff.request().ptr;

    for (int i = 0; i < n; ++i) {
        double wa = w_ace_avg_ptr[i], la = l_ace_avg_ptr[i];
        double wd = w_df_avg_ptr[i],  ld = l_df_avg_ptr[i];
        double wb = w_bp_avg_ptr[i],  lb = l_bp_avg_ptr[i];

        double ad = (is_nan(wa) || is_nan(la)) ? numeric_limits<double>::quiet_NaN() : (wa - la);
        double dd = (is_nan(wd) || is_nan(ld)) ? numeric_limits<double>::quiet_NaN() : -(wd - ld);
        double bd = (is_nan(wb) || is_nan(lb)) ? numeric_limits<double>::quiet_NaN() : (wb - lb);

        ace_diff_ptr[i] = round_half_even(ad, 5);
        df_diff_ptr[i]  = round_half_even(dd, 5);
        bp_diff_ptr[i]  = round_half_even(bd, 5);
    }

    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double> dt = t1 - t0;

    dict out;
    out["w_ace_avg"] = w_ace_avg;
    out["l_ace_avg"] = l_ace_avg;
    out["w_df_avg"]  = w_df_avg;
    out["l_df_avg"]  = l_df_avg;
    out["w_bpSaved_avg"] = w_bp_avg;
    out["l_bpSaved_avg"] = l_bp_avg;
    out["ace_diff"] = ace_diff;
    out["df_diff"] = df_diff;
    out["bp_diff"] = bp_diff;
    out["wall_time_sec"] = dt.count();

    #ifdef _OPENMP
    out["omp_threads"] = (num_threads > 0 ? num_threads : omp_get_max_threads());
    #else
    out["omp_threads"] = 1;
    #endif

    return out;
}

PYBIND11_MODULE(native_rolling, m) {
    m.doc() = "High-performance rolling feature computation for tennis dataset (per-player windowed means with lookback), parallelized with OpenMP.";
    m.def("compute_rolling_features", &compute_rolling_features,
        arg("player_ids"), arg("opponent_ids"),
        arg("w_ace"), arg("l_ace"),
        arg("w_df"), arg("l_df"),
        arg("w_bpSaved"), arg("l_bpSaved"),
        arg("window") = 10,
        arg("lookback") = 600,
        arg("num_threads") = -1
    );
}
