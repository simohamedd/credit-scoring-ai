package com.simo.credit;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/** Wraps the decision tree as a lending decision: approve, refer to a human, or decline. */
public class CreditScorer {

    public enum Outcome { APPROVE, REFER, DECLINE }

    public record Result(Outcome outcome, double repayChance, List<String> reasons) { }

    /**
     * How one outcome band of a policy looks on a labelled set of applicants.
     *
     * @param applicants how many applicants got this outcome
     * @param repaid     how many of them actually repaid
     * @param share      their share of all applicants assessed
     */
    public record Band(int applicants, int repaid, double share) {
        /** Share of this band that repaid, or NaN when nobody landed in it. */
        public double repaidRate() {
            return applicants == 0 ? Double.NaN : (double) repaid / applicants;
        }
    }

    /** Default policy: approve from a 75% repay chance, refer to a human from 50%, decline below. */
    public static final double DEFAULT_APPROVE_AT = 0.75;
    public static final double DEFAULT_REFER_AT = 0.5;

    private final DecisionTree tree;
    private final double approveAt;
    private final double referAt;

    private CreditScorer(DecisionTree tree, double approveAt, double referAt) {
        if (!(referAt >= 0 && approveAt <= 1 && referAt <= approveAt))
            throw new IllegalArgumentException("Cut-offs must satisfy 0 <= refer <= approve <= 1, got approve "
                    + approveAt + " and refer " + referAt);
        this.tree = tree;
        this.approveAt = approveAt;
        this.referAt = referAt;
    }

    public static CreditScorer train(List<Applicant> history, int maxDepth) {
        double[][] x = history.stream().map(Applicant::features).toArray(double[][]::new);
        boolean[] y = new boolean[history.size()];
        for (int i = 0; i < y.length; i++) y[i] = history.get(i).repaid();
        int minLeaf = Math.max(10, history.size() / 200);
        return new CreditScorer(new DecisionTree(maxDepth, minLeaf).fit(x, y, Applicant.FEATURES),
                DEFAULT_APPROVE_AT, DEFAULT_REFER_AT);
    }

    /**
     * The same trained model with a different lending policy: approve when the repay chance is at least
     * {@code approveAt}, refer to a human when it is at least {@code referAt}, otherwise decline.
     * Setting both to the same value removes the referral band. This scorer is left unchanged.
     */
    public CreditScorer withThresholds(double approveAt, double referAt) {
        return new CreditScorer(tree, approveAt, referAt);
    }

    /** Saves the trained model and this scorer's cut-offs to a text file, so it can be loaded without retraining. */
    public void save(Path file) throws IOException {
        try (BufferedWriter out = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            out.write(FORMAT + "\n");
            out.write("policy " + approveAt + " " + referAt + "\n");
            tree.write(out);
        }
    }

    /**
     * Loads a scorer saved with {@link #save}. It makes the same decisions, with the same explanations,
     * as the scorer that was saved. Files that are corrupt, truncated or built for different applicant
     * details are refused with an {@link IOException}.
     */
    public static CreditScorer load(Path file) throws IOException {
        try (BufferedReader in = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            DecisionTree.LineReader lines = new DecisionTree.LineReader(in);
            if (!FORMAT.equals(lines.next())) throw lines.error("not a credit scorer file (expected \"" + FORMAT + "\")");
            String[] policy = lines.fields("policy", 2);
            double approve = lines.number(policy[1], 0, 1);
            double refer = lines.number(policy[2], 0, 1);
            DecisionTree tree = DecisionTree.read(lines);
            if (!Arrays.equals(tree.featureNames(), Applicant.FEATURES))
                throw new IOException("Model was trained on different applicant details: "
                        + Arrays.toString(tree.featureNames()));
            try {
                return new CreditScorer(tree, approve, refer);
            } catch (IllegalArgumentException e) {
                throw new IOException("Invalid model: " + e.getMessage(), e);
            }
        }
    }

    private static final String FORMAT = "credit-scorer v1";

    public double approveAt() {
        return approveAt;
    }

    public double referAt() {
        return referAt;
    }

    public Result assess(Applicant a) {
        DecisionTree.Decision d = tree.explain(a.features());
        double p = tree.predictProbability(a.features());
        Outcome o = p >= approveAt ? Outcome.APPROVE : p >= referAt ? Outcome.REFER : Outcome.DECLINE;
        return new Result(o, p, d.reasons());
    }

    /**
     * Applies this scorer's policy to applicants whose outcome is known and summarises each band:
     * how many were approved, referred and declined, and how often each group really repaid.
     * Every outcome is present in the result, in APPROVE, REFER, DECLINE order.
     */
    public Map<Outcome, Band> policyReport(List<Applicant> data) {
        requireData(data);
        int[] count = new int[Outcome.values().length];
        int[] repaid = new int[count.length];
        for (Applicant a : data) {
            int o = assess(a).outcome().ordinal();
            count[o]++;
            if (a.repaid()) repaid[o]++;
        }
        Map<Outcome, Band> report = new EnumMap<>(Outcome.class);
        for (Outcome o : Outcome.values())
            report.put(o, new Band(count[o.ordinal()], repaid[o.ordinal()], (double) count[o.ordinal()] / data.size()));
        return report;
    }

    public double accuracy(List<Applicant> data) {
        requireData(data);
        long correct = data.stream().filter(a -> (tree.predictProbability(a.features()) >= 0.5) == a.repaid()).count();
        return (double) correct / data.size();
    }

    private static void requireData(List<Applicant> data) {
        if (data == null || data.isEmpty()) throw new IllegalArgumentException("Need at least one applicant to evaluate on");
    }

    public DecisionTree tree() {
        return tree;
    }
}
