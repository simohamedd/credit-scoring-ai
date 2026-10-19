package com.simo.credit;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates applicants whose chance of repaying depends on their finances through a hidden scoring rule plus noise.
 * The model never sees the rule; it has to rediscover it from examples.
 */
public final class SyntheticApplicants {
    private SyntheticApplicants() {
    }

    public static List<Applicant> generate(int n, long seed) {
        Random r = new Random(seed);
        List<Applicant> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double income = Math.round(Math.exp(10.3 + 0.45 * r.nextGaussian()));
            double dti = clamp(0.28 + 0.14 * r.nextGaussian(), 0, 0.95);
            double history = Math.max(0, Math.round((6 + 5 * r.nextGaussian()) * 10) / 10.0);
            int missed = r.nextDouble() < 0.7 ? 0 : 1 + r.nextInt(5);
            double loan = Math.round(income * clamp(0.3 + 0.25 * r.nextGaussian(), 0.02, 1.5) / 100) * 100.0;

            double risk = 3.2 * (dti - 0.3) + 0.55 * missed + 1.8 * (loan / income - 0.3)
                    - 0.08 * Math.min(history, 15) - 0.6 * Math.log(income / 30000);
            double pRepay = 1 / (1 + Math.exp(3 * (risk - 0.2)));
            out.add(new Applicant(income, round2(dti), history, missed, loan, r.nextDouble() < pRepay));
        }
        return out;
    }

    private static double clamp(double x, double lo, double hi) { return Math.max(lo, Math.min(hi, x)); }
    private static double round2(double x) { return Math.round(x * 100) / 100.0; }
}
