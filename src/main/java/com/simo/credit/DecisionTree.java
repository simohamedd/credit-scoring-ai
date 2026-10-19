package com.simo.credit;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A CART decision tree classifier using Gini impurity, built from scratch.
 * Unlike a black-box model, every prediction comes with the exact rules that led to it.
 */
public class DecisionTree {

    private final int maxDepth;
    private final int minSamplesLeaf;
    private String[] featureNames;
    private Node root;

    /** A node is either a split ("feature <= threshold") or a leaf holding the share of positive examples. */
    private static final class Node {
        int feature = -1;
        double threshold;
        Node left, right;
        double positiveRate;
        int samples;

        boolean isLeaf() { return feature < 0; }
    }

    public record Decision(boolean positive, double confidence, List<String> reasons) { }

    public DecisionTree(int maxDepth, int minSamplesLeaf) {
        if (maxDepth < 1 || minSamplesLeaf < 1) throw new IllegalArgumentException("Depth and leaf size must be at least 1");
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
    }

    public DecisionTree fit(double[][] x, boolean[] y, String[] featureNames) {
        this.featureNames = featureNames;
        int[] all = new int[x.length];
        for (int i = 0; i < all.length; i++) all[i] = i;
        root = build(x, y, all, 0);
        return this;
    }

    private Node build(double[][] x, boolean[] y, int[] rows, int depth) {
        Node node = new Node();
        node.samples = rows.length;
        int pos = 0;
        for (int i : rows) if (y[i]) pos++;
        node.positiveRate = (double) pos / rows.length;
        if (depth >= maxDepth || pos == 0 || pos == rows.length || rows.length < 2 * minSamplesLeaf) return node;

        double bestScore = gini(pos, rows.length);
        int bestFeature = -1;
        double bestThreshold = 0;
        for (int f = 0; f < x[0].length; f++) {
            final int feat = f;
            Integer[] sorted = Arrays.stream(rows).boxed().toArray(Integer[]::new);
            Arrays.sort(sorted, (a, b) -> Double.compare(x[a][feat], x[b][feat]));
            int leftPos = 0;
            for (int k = 0; k < sorted.length - 1; k++) {
                if (y[sorted[k]]) leftPos++;
                int leftN = k + 1, rightN = sorted.length - leftN;
                double here = x[sorted[k]][f], next = x[sorted[k + 1]][f];
                if (here == next || leftN < minSamplesLeaf || rightN < minSamplesLeaf) continue;
                double score = (leftN * gini(leftPos, leftN) + rightN * gini(pos - leftPos, rightN)) / sorted.length;
                if (score < bestScore - 1e-12) {
                    bestScore = score;
                    bestFeature = f;
                    bestThreshold = (here + next) / 2;
                }
            }
        }
        if (bestFeature < 0) return node;

        final int bf = bestFeature;
        final double bt = bestThreshold;
        node.feature = bf;
        node.threshold = bt;
        node.left = build(x, y, Arrays.stream(rows).filter(i -> x[i][bf] <= bt).toArray(), depth + 1);
        node.right = build(x, y, Arrays.stream(rows).filter(i -> x[i][bf] > bt).toArray(), depth + 1);
        return node;
    }

    static double gini(int pos, int n) {
        if (n == 0) return 0;
        double p = (double) pos / n;
        return 2 * p * (1 - p);
    }

    public double predictProbability(double[] row) {
        Node n = root();
        while (!n.isLeaf()) n = row[n.feature] <= n.threshold ? n.left : n.right;
        return n.positiveRate;
    }

    /** The prediction plus the human-readable rules followed to reach it. */
    public Decision explain(double[] row) {
        Node n = root();
        List<String> reasons = new ArrayList<>();
        while (!n.isLeaf()) {
            boolean left = row[n.feature] <= n.threshold;
            reasons.add(String.format("%s is %s %s (applicant: %s)", featureNames[n.feature],
                    left ? "at most" : "above", fmt(n.threshold), fmt(row[n.feature])));
            n = left ? n.left : n.right;
        }
        boolean positive = n.positiveRate >= 0.5;
        return new Decision(positive, positive ? n.positiveRate : 1 - n.positiveRate, reasons);
    }

    public int depth() {
        return depth(root());
    }

    public int leaves() {
        return leaves(root());
    }

    /** Prints the tree as indented rules. */
    public String describe() {
        return describe(Integer.MAX_VALUE);
    }

    /** Prints the top levels of the tree; deeper branches are summarised. */
    public String describe(int maxLevels) {
        StringBuilder sb = new StringBuilder();
        describe(root(), "", sb, maxLevels);
        return sb.toString();
    }

    private void describe(Node n, String indent, StringBuilder sb, int levelsLeft) {
        if (!n.isLeaf() && levelsLeft == 0) {
            sb.append(indent).append(String.format("-> repay chance %.0f%% (%d applicants, %d more rules below)%n",
                    100 * n.positiveRate, n.samples, leaves(n) - 1));
            return;
        }
        if (n.isLeaf()) {
            sb.append(indent).append(String.format("-> repay chance %.0f%% (%d applicants)%n", 100 * n.positiveRate, n.samples));
            return;
        }
        sb.append(indent).append("if ").append(featureNames[n.feature]).append(" <= ").append(fmt(n.threshold)).append(":\n");
        describe(n.left, indent + "    ", sb, levelsLeft - 1);
        sb.append(indent).append("else:\n");
        describe(n.right, indent + "    ", sb, levelsLeft - 1);
    }

    private Node root() {
        if (root == null) throw new IllegalStateException("Call fit() first");
        return root;
    }

    private static int depth(Node n) { return n.isLeaf() ? 0 : 1 + Math.max(depth(n.left), depth(n.right)); }
    private static int leaves(Node n) { return n.isLeaf() ? 1 : leaves(n.left) + leaves(n.right); }

    private static String fmt(double v) {
        if (Math.abs(v) >= 1000) return String.format("%,.0f", v);
        if (v == Math.rint(v)) return String.valueOf((long) v);
        return String.format("%.2f", v);
    }
}
