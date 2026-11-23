package com.simo.credit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

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
    void repeatedSplitsOnAFeatureBecomeOneRange() {
        // Only the middle band of x is positive, so the tree has to split on x twice.
        double[][] x = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}};
        boolean[] y = {false, false, false, true, true, true, false, false, false};
        DecisionTree t = new DecisionTree(3, 1).fit(x, y, XY);
        DecisionTree.Decision d = t.explain(new double[] {5, 0});
        assertTrue(d.positive());
        assertEquals(List.of("x is above 3.50 and at most 6.50 (applicant: 5)"), d.reasons());
        assertEquals(List.of("x is at most 3.50 (applicant: 2)"), t.explain(new double[] {2, 0}).reasons());
    }

    @Test
    void explanationsNameEachFeatureOnce() {
        CreditScorer s = CreditScorer.train(SyntheticApplicants.generate(6_000, 7), 6);
        for (Applicant a : SyntheticApplicants.generate(300, 8)) {
            List<String> reasons = s.assess(a).reasons();
            long distinct = reasons.stream().map(r -> r.substring(0, r.indexOf(" is "))).distinct().count();
            assertEquals(reasons.size(), distinct, reasons.toString());
        }
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
    void lenderCanSetTheirOwnCutOffs() {
        CreditScorer s = CreditScorer.train(SyntheticApplicants.generate(10_000, 3), 4);
        assertEquals(0.75, s.approveAt(), 1e-9);
        assertEquals(0.5, s.referAt(), 1e-9);

        Applicant a = new Applicant(60_000, 0.12, 12, 0, 5_000, false);
        double p = s.assess(a).repayChance();
        assertTrue(p > 0.75 && p < 1, "repay chance " + p);

        CreditScorer strict = s.withThresholds(1, 0.5);
        assertEquals(CreditScorer.Outcome.REFER, strict.assess(a).outcome());
        assertEquals(p, strict.assess(a).repayChance(), 1e-12);
        assertEquals(CreditScorer.Outcome.APPROVE, s.assess(a).outcome(), "original scorer must be unchanged");

        assertEquals(CreditScorer.Outcome.APPROVE, s.withThresholds(p, p).assess(a).outcome());
        assertEquals(CreditScorer.Outcome.REFER, s.withThresholds(1, p).assess(a).outcome());
        assertEquals(CreditScorer.Outcome.DECLINE, s.withThresholds(1, Math.nextUp(p)).assess(a).outcome());
    }

    @Test
    void rejectsImpossibleCutOffs() {
        CreditScorer s = CreditScorer.train(SyntheticApplicants.generate(2_000, 6), 2);
        assertThrows(IllegalArgumentException.class, () -> s.withThresholds(0.5, 0.75));
        assertThrows(IllegalArgumentException.class, () -> s.withThresholds(1.2, 0.5));
        assertThrows(IllegalArgumentException.class, () -> s.withThresholds(0.75, -0.1));
        assertThrows(IllegalArgumentException.class, () -> s.withThresholds(Double.NaN, 0.5));
        assertThrows(IllegalArgumentException.class, () -> s.withThresholds(0.75, Double.NaN));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> s.withThresholds(0.5, 0.75));
        assertTrue(e.getMessage().contains("refer <= approve"), e.getMessage());
        s.withThresholds(0, 0);
        s.withThresholds(1, 1);
    }

    @Test
    void policyReportCountsEachBandAndItsRepaymentRate() {
        List<Applicant> data = SyntheticApplicants.generate(10_000, 3);
        CreditScorer s = CreditScorer.train(data.subList(0, 7_000), 4);
        List<Applicant> test = data.subList(7_000, 10_000);
        Map<CreditScorer.Outcome, CreditScorer.Band> report = s.policyReport(test);

        assertEquals(List.of(CreditScorer.Outcome.values()), List.copyOf(report.keySet()));
        assertEquals(test.size(), report.values().stream().mapToInt(CreditScorer.Band::applicants).sum());
        assertEquals(1.0, report.values().stream().mapToDouble(CreditScorer.Band::share).sum(), 1e-9);
        long approved = test.stream().filter(a -> s.assess(a).outcome() == CreditScorer.Outcome.APPROVE).count();
        assertEquals(approved, report.get(CreditScorer.Outcome.APPROVE).applicants());

        double approveRate = report.get(CreditScorer.Outcome.APPROVE).repaidRate();
        double declineRate = report.get(CreditScorer.Outcome.DECLINE).repaidRate();
        assertTrue(approveRate > declineRate + 0.3, "approved " + approveRate + " vs declined " + declineRate);

        CreditScorer.Band strictApprove = s.withThresholds(0.9, 0.5).policyReport(test).get(CreditScorer.Outcome.APPROVE);
        assertTrue(strictApprove.applicants() <= approved);
    }

    @Test
    void policyReportHandlesEmptyBandsAndRefusesEmptyData() {
        CreditScorer s = CreditScorer.train(SyntheticApplicants.generate(2_000, 6), 2);
        List<Applicant> few = SyntheticApplicants.generate(50, 9);
        CreditScorer.Band refer = s.withThresholds(0.6, 0.6).policyReport(few).get(CreditScorer.Outcome.REFER);
        assertEquals(0, refer.applicants());
        assertEquals(0.0, refer.share(), 1e-9);
        assertTrue(Double.isNaN(refer.repaidRate()));

        assertThrows(IllegalArgumentException.class, () -> s.policyReport(List.of()));
        assertThrows(IllegalArgumentException.class, () -> s.accuracy(List.of()));
    }

    @Test
    void crossValidationPicksADepthThatBeatsAStump() {
        List<Applicant> data = SyntheticApplicants.generate(4_000, 4);
        CrossValidation.Result r = CrossValidation.chooseDepth(data, 4, 1, 5);
        assertTrue(r.bestDepth() >= 2 && r.bestDepth() <= 5, "depth " + r.bestDepth());
        assertTrue(r.meanAccuracyByDepth().get(r.bestDepth()) > r.meanAccuracyByDepth().get(1));
        assertThrows(IllegalArgumentException.class, () -> CrossValidation.meanAccuracy(data, 1, 3));
    }

    @Test
    void crossValidationRefusesBadArguments() {
        List<Applicant> data = SyntheticApplicants.generate(200, 4);
        IllegalArgumentException range = assertThrows(IllegalArgumentException.class,
                () -> CrossValidation.chooseDepth(data, 4, 5, 3));
        assertTrue(range.getMessage().contains("5 to 3"), range.getMessage());
        assertThrows(IllegalArgumentException.class, () -> CrossValidation.chooseDepth(data, 4, 0, 3));
        assertThrows(IllegalArgumentException.class, () -> CrossValidation.meanAccuracy(data, 4, 0));
        assertThrows(IllegalArgumentException.class, () -> CrossValidation.meanAccuracy(data, 201, 2));
        assertThrows(IllegalArgumentException.class, () -> CrossValidation.meanAccuracy(List.of(), 2, 2));
        assertThrows(IllegalArgumentException.class, () -> CrossValidation.meanAccuracy(null, 2, 2));
        assertEquals(1, CrossValidation.chooseDepth(data, 4, 1, 1).meanAccuracyByDepth().size());
    }

    @Test
    void predictionRefusesRowsThatDontMatchTheTrainingFeatures() {
        double[][] x = {{1, 0}, {2, 0}, {3, 0}, {7, 0}, {8, 0}, {9, 0}};
        boolean[] y = {false, false, false, true, true, true};
        DecisionTree t = new DecisionTree(3, 1).fit(x, y, XY);

        IllegalArgumentException shortRow = assertThrows(IllegalArgumentException.class,
                () -> t.predictProbability(new double[] {6}));
        assertTrue(shortRow.getMessage().contains("Expected 2 values"), shortRow.getMessage());
        assertThrows(IllegalArgumentException.class, () -> t.predictProbability(new double[] {6, 0, 1}));
        assertThrows(IllegalArgumentException.class, () -> t.predictProbability(null));
        assertThrows(IllegalArgumentException.class, () -> t.explain(new double[] {6}));

        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                () -> t.predictProbability(new double[] {Double.NaN, 0}));
        assertTrue(missing.getMessage().contains("x"), missing.getMessage());
        assertThrows(IllegalArgumentException.class, () -> t.explain(new double[] {6, Double.POSITIVE_INFINITY}));
        assertThrows(IllegalStateException.class, () -> new DecisionTree(2, 1).predictProbability(new double[] {1, 2}));
    }

    @Test
    void rejectsImpossibleApplicants() {
        assertThrows(IllegalArgumentException.class, () -> new Applicant(0, 0.2, 5, 0, 5_000, false));
        assertThrows(IllegalArgumentException.class, () -> new Applicant(-30_000, 0.2, 5, 0, 5_000, false));
        assertThrows(IllegalArgumentException.class, () -> new Applicant(Double.NaN, 0.2, 5, 0, 5_000, false));
        assertThrows(IllegalArgumentException.class, () -> new Applicant(30_000, 1.2, 5, 0, 5_000, false));
        assertThrows(IllegalArgumentException.class, () -> new Applicant(30_000, Double.NaN, 5, 0, 5_000, false));
        assertThrows(IllegalArgumentException.class, () -> new Applicant(30_000, 0.2, -1, 0, 5_000, false));
        assertThrows(IllegalArgumentException.class, () -> new Applicant(30_000, 0.2, 5, -2, 5_000, false));
        assertThrows(IllegalArgumentException.class, () -> new Applicant(30_000, 0.2, 5, 0, -100, false));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new Applicant(0, 0.2, 5, 0, 5_000, false));
        assertTrue(e.getMessage().contains("Annual income"), e.getMessage());
    }

    @Test
    void acceptsBoundaryApplicants() {
        Applicant a = new Applicant(1, 0, 0, 0, 0, false);
        assertEquals(0.0, a.features()[4], 1e-9);
        new Applicant(30_000, 1, 0, 0, 1_000, true);
        SyntheticApplicants.generate(20_000, 99);
    }

    @Test
    void rejectsMalformedTrainingData() {
        DecisionTree t = new DecisionTree(2, 1);
        assertThrows(IllegalArgumentException.class, () -> t.fit(new double[0][], new boolean[0], XY));
        assertThrows(IllegalArgumentException.class,
                () -> t.fit(new double[][] {{1, 2}, {3, 4}}, new boolean[] {true}, XY));
        assertThrows(IllegalArgumentException.class,
                () -> t.fit(new double[][] {{1, 2}, {3}}, new boolean[] {true, false}, XY));
        assertThrows(IllegalArgumentException.class,
                () -> t.fit(new double[][] {{1, 2}, {3, 4}}, new boolean[] {true, false}, new String[] {"x"}));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> t.fit(new double[][] {{1, 2}, {Double.NaN, 4}}, new boolean[] {true, false}, XY));
        assertTrue(e.getMessage().contains("Row 1") && e.getMessage().contains("x"), e.getMessage());
    }

    @Test
    void featureImportanceCreditsOnlyTheFeaturesUsed() {
        double[][] x = {{1, 5}, {2, 3}, {3, 9}, {7, 4}, {8, 8}, {9, 2}};
        boolean[] y = {false, false, false, true, true, true};
        Map<String, Double> imp = new DecisionTree(3, 1).fit(x, y, XY).featureImportances();
        assertEquals(List.of("x", "y"), List.copyOf(imp.keySet()));
        assertEquals(1.0, imp.get("x"), 1e-9);
        assertEquals(0.0, imp.get("y"), 1e-9);
    }

    @Test
    void featureImportancesSumToOneAndAreRankedOnRealisticData() {
        Map<String, Double> imp = CreditScorer.train(SyntheticApplicants.generate(5_000, 5), 4).tree().featureImportances();
        assertEquals(Applicant.FEATURES.length, imp.size());
        assertEquals(1.0, imp.values().stream().mapToDouble(Double::doubleValue).sum(), 1e-9);
        double previous = Double.MAX_VALUE;
        for (double share : imp.values()) {
            assertTrue(share >= 0 && share <= previous, imp.toString());
            previous = share;
        }
    }

    @Test
    void unsplitTreeHasZeroImportanceAndUnfittedTreeRefuses() {
        double[][] x = {{1, 2}, {3, 4}, {5, 6}};
        Map<String, Double> imp = new DecisionTree(3, 1).fit(x, new boolean[] {true, true, true}, XY).featureImportances();
        assertEquals(0.0, imp.get("x"), 1e-9);
        assertEquals(0.0, imp.get("y"), 1e-9);
        assertThrows(IllegalStateException.class, () -> new DecisionTree(2, 1).featureImportances());
    }

    @Test
    void savedScorerMakesTheSameDecisions(@TempDir Path dir) throws IOException {
        CreditScorer s = CreditScorer.train(SyntheticApplicants.generate(6_000, 12), 5).withThresholds(0.8, 0.55);
        Path file = dir.resolve("model.txt");
        s.save(file);
        CreditScorer loaded = CreditScorer.load(file);

        assertEquals(0.8, loaded.approveAt(), 0);
        assertEquals(0.55, loaded.referAt(), 0);
        assertEquals(s.tree().describe(), loaded.tree().describe());
        assertEquals(s.tree().featureImportances(), loaded.tree().featureImportances());
        for (Applicant a : SyntheticApplicants.generate(500, 13)) {
            CreditScorer.Result before = s.assess(a), after = loaded.assess(a);
            assertEquals(before.outcome(), after.outcome());
            assertEquals(before.repayChance(), after.repayChance(), 0);
            assertEquals(before.reasons(), after.reasons());
        }
    }

    @Test
    void refusesCorruptOrForeignModelFiles(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("model.txt");
        CreditScorer.train(SyntheticApplicants.generate(2_000, 6), 3).save(file);
        List<String> good = Files.readAllLines(file);

        Files.write(file, good.subList(0, good.size() - 1));
        IOException truncated = assertThrows(IOException.class, () -> CreditScorer.load(file));
        assertTrue(truncated.getMessage().contains("ends before"), truncated.getMessage());

        Files.writeString(file, "hello\n");
        assertThrows(IOException.class, () -> CreditScorer.load(file));

        List<String> badRate = new java.util.ArrayList<>(good);
        badRate.set(badRate.size() - 1, "leaf 10 1.5");
        Files.write(file, badRate);
        IOException e = assertThrows(IOException.class, () -> CreditScorer.load(file));
        assertTrue(e.getMessage().contains("line " + good.size()), e.getMessage());

        List<String> extra = new java.util.ArrayList<>(good);
        extra.add("leaf 1 0.5");
        Files.write(file, extra);
        assertThrows(IOException.class, () -> CreditScorer.load(file));

        Path other = dir.resolve("other.txt");
        double[][] x = {{1, 0}, {2, 0}, {8, 0}, {9, 0}};
        DecisionTree t = new DecisionTree(2, 1).fit(x, new boolean[] {false, false, true, true}, XY);
        try (java.io.Writer w = Files.newBufferedWriter(other)) {
            w.write("credit-scorer v1\npolicy 0.75 0.5\n");
            t.write(w);
        }
        IOException foreign = assertThrows(IOException.class, () -> CreditScorer.load(other));
        assertTrue(foreign.getMessage().contains("different applicant details"), foreign.getMessage());
    }

    @Test
    void treeRoundTripsThroughText() throws IOException {
        double[][] x = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}};
        boolean[] y = {false, false, false, true, true, true, false, false, false};
        DecisionTree t = new DecisionTree(3, 1).fit(x, y, XY);
        java.io.StringWriter out = new java.io.StringWriter();
        t.write(out);
        DecisionTree back = DecisionTree.read(new java.io.BufferedReader(new java.io.StringReader(out.toString())));
        assertEquals(t.describe(), back.describe());
        assertEquals(t.explain(new double[] {5, 0}).reasons(), back.explain(new double[] {5, 0}).reasons());
        assertThrows(IllegalStateException.class, () -> new DecisionTree(2, 1).write(new java.io.StringWriter()));
    }
}
