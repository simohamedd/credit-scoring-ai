package com.simo.credit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class RankingTest {

    @Test
    void perfectReversedAndUselessRankings() {
        boolean[] y = {false, false, true, true};
        assertEquals(1.0, CreditScorer.rocAuc(new double[] {0.1, 0.2, 0.8, 0.9}, y), 1e-12);
        assertEquals(0.0, CreditScorer.rocAuc(new double[] {0.9, 0.8, 0.2, 0.1}, y), 1e-12);
        assertEquals(0.5, CreditScorer.rocAuc(new double[] {0.6, 0.6, 0.6, 0.6}, y), 1e-12);
    }

    @Test
    void tiesCountAsHalfWhateverTheOrder() {
        // Pairs (repaid, defaulted): 0.7>0.3, 0.7>0.5, 0.5=0.5 (half), 0.5>0.3 -> 3.5 of 4
        double[] s = {0.5, 0.3, 0.7, 0.5};
        boolean[] y = {true, false, true, false};
        assertEquals(3.5 / 4, CreditScorer.rocAuc(s, y), 1e-12);
        assertEquals(3.5 / 4, CreditScorer.rocAuc(new double[] {0.5, 0.7, 0.3, 0.5}, new boolean[] {false, true, false, true}), 1e-12);
    }

    @Test
    void matchesCountingEveryPairOnRealPredictions() {
        List<Applicant> all = SyntheticApplicants.generate(3_000, 4);
        CreditScorer scorer = CreditScorer.train(all.subList(0, 2_000), 4);
        List<Applicant> test = all.subList(2_000, 3_000);

        double wins = 0;
        long pairs = 0;
        for (Applicant good : test) {
            if (!good.repaid()) continue;
            double g = scorer.assess(good).repayChance();
            for (Applicant bad : test) {
                if (bad.repaid()) continue;
                double b = scorer.assess(bad).repayChance();
                wins += g > b ? 1 : g == b ? 0.5 : 0;
                pairs++;
            }
        }
        double auc = scorer.rocAuc(test);
        assertEquals(wins / pairs, auc, 1e-9);
        assertEquals(2 * auc - 1, scorer.gini(test), 1e-12);
        assertTrue(auc > 0.7, "the model should rank much better than chance, got " + auc);
    }

    @Test
    void needsBothOutcomes() {
        List<Applicant> all = SyntheticApplicants.generate(1_000, 4);
        CreditScorer scorer = CreditScorer.train(all, 3);
        List<Applicant> repaidOnly = all.stream().filter(Applicant::repaid).toList();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> scorer.rocAuc(repaidOnly));
        assertTrue(e.getMessage().contains("0 defaulted"), e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> scorer.gini(List.of()));
    }
}
