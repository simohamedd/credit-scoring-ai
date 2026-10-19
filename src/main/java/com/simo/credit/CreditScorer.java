package com.simo.credit;

import java.util.List;

/** Wraps the decision tree as a lending decision: approve, refer to a human, or decline. */
public class CreditScorer {

    public enum Outcome { APPROVE, REFER, DECLINE }

    public record Result(Outcome outcome, double repayChance, List<String> reasons) { }

    private final DecisionTree tree;

    private CreditScorer(DecisionTree tree) {
        this.tree = tree;
    }

    public static CreditScorer train(List<Applicant> history, int maxDepth) {
        double[][] x = history.stream().map(Applicant::features).toArray(double[][]::new);
        boolean[] y = new boolean[history.size()];
        for (int i = 0; i < y.length; i++) y[i] = history.get(i).repaid();
        int minLeaf = Math.max(10, history.size() / 200);
        return new CreditScorer(new DecisionTree(maxDepth, minLeaf).fit(x, y, Applicant.FEATURES));
    }

    public Result assess(Applicant a) {
        DecisionTree.Decision d = tree.explain(a.features());
        double p = tree.predictProbability(a.features());
        Outcome o = p >= 0.75 ? Outcome.APPROVE : p >= 0.5 ? Outcome.REFER : Outcome.DECLINE;
        return new Result(o, p, d.reasons());
    }

    public double accuracy(List<Applicant> data) {
        long correct = data.stream().filter(a -> (tree.predictProbability(a.features()) >= 0.5) == a.repaid()).count();
        return (double) correct / data.size();
    }

    public DecisionTree tree() {
        return tree;
    }
}
