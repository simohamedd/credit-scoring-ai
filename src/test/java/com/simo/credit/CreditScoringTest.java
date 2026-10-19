package com.simo.credit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class CreditScoringTest {

    private static final String[] XY = {"x", "y"};

    @Test
    void giniImpurity() {
        assertEquals(0.0, DecisionTree.gini(0, 10), 1e-9);
        assertEquals(0.0, DecisionTree.gini(10, 10), 1e-9);
        assertEquals(0.5, DecisionTree.gini(5, 10), 1e-9);
    }

    @Test
    void learnsASimpleThreshold() {
        double[][] x = {{1, 0}, {2, 0}, {3, 0}, {7, 0}, {8, 0}, {9, 0}};
        boolean[] y = {false, false, false, true, true, true};
        DecisionTree t = new DecisionTree(3, 1).fit(x, y, XY);
        assertEquals(1, t.depth());
        assertEquals(1.0, t.predictProbability(new double[] {6, 0}), 1e-9);
        assertEquals(0.0, t.predictProbability(new double[] {4, 0}), 1e-9);
        assertTrue(t.explain(new double[] {6, 0}).reasons().get(0).startsWith("x is above 5"));
    }

    @Test
    void learnsAnInteractionThatNeedsTwoLevels() {
        double[][] x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 1}, {1, 0}};
        boolean[] y = {false, false, false, true, false, true, false, false};
        DecisionTree t = new DecisionTree(2, 1).fit(x, y, XY);
        assertTrue(t.predictProbability(new double[] {1, 1}) > 0.9);
        assertTrue(t.predictProbability(new double[] {1, 0}) < 0.1);
        assertEquals(2, t.explain(new double[] {1, 1}).reasons().size());
    }

    @Test
    void respectsMaxDepth() {
        List<Applicant> data = SyntheticApplicants.generate(3_000, 1);
        assertTrue(CreditScorer.train(data, 3).tree().depth() <= 3);
        assertThrows(IllegalArgumentException.class, () -> new DecisionTree(0, 1));
    }

    @Test
    void beatsTheBaselineOnUnseenApplicants() {
        List<Applicant> all = SyntheticApplicants.generate(10_000, 2);
        List<Applicant> test = all.subList(7_000, 10_000);
        long repaid = test.stream().filter(Applicant::repaid).count();
        double baseline = Math.max(repaid, test.size() - repaid) / (double) test.size();
        double acc = CreditScorer.train(all.subList(0, 7_000), 4).accuracy(test);
        assertTrue(acc > baseline + 0.05, "accuracy " + acc + " vs baseline " + baseline);
    }

    @Test
    void decisionsAreSensibleAndExplained() {
        CreditScorer s = CreditScorer.train(SyntheticApplicants.generate(10_000, 3), 4);
        CreditScorer.Result good = s.assess(new Applicant(60_000, 0.12, 12, 0, 5_000, false));
        CreditScorer.Result bad = s.assess(new Applicant(20_000, 0.7, 1, 5, 18_000, false));
        assertEquals(CreditScorer.Outcome.APPROVE, good.outcome());
        assertEquals(CreditScorer.Outcome.DECLINE, bad.outcome());
        assertFalse(bad.reasons().isEmpty());
    }

    @Test
    void crossValidationPicksADepthThatBeatsAStump() {
        List<Applicant> data = SyntheticApplicants.generate(4_000, 4);
        CrossValidation.Result r = CrossValidation.chooseDepth(data, 4, 1, 5);
        assertTrue(r.bestDepth() >= 2 && r.bestDepth() <= 5, "depth " + r.bestDepth());
        assertTrue(r.meanAccuracyByDepth().get(r.bestDepth()) > r.meanAccuracyByDepth().get(1));
        assertThrows(IllegalArgumentException.class, () -> CrossValidation.meanAccuracy(data, 1, 3));
    }
}
