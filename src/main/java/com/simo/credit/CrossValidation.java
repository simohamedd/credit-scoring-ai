package com.simo.credit;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * k-fold cross-validation: splits the data into k parts, trains on k-1 and tests on the last, k times.
 * Used to choose the tree depth from the data instead of guessing it.
 */
public final class CrossValidation {

    public record Result(int bestDepth, Map<Integer, Double> meanAccuracyByDepth) { }

    private CrossValidation() {
    }

    public static double meanAccuracy(List<Applicant> data, int folds, int depth) {
        if (folds < 2 || folds > data.size()) throw new IllegalArgumentException("Folds must be between 2 and the number of applicants");
        double total = 0;
        for (int f = 0; f < folds; f++) {
            List<Applicant> train = new ArrayList<>(), test = new ArrayList<>();
            for (int i = 0; i < data.size(); i++) (i % folds == f ? test : train).add(data.get(i));
            total += CreditScorer.train(train, depth).accuracy(test);
        }
        return total / folds;
    }

    /** Tries each depth and returns the one with the best average accuracy (the shallowest on a tie). */
    public static Result chooseDepth(List<Applicant> data, int folds, int minDepth, int maxDepth) {
        Map<Integer, Double> scores = new LinkedHashMap<>();
        int best = minDepth;
        for (int d = minDepth; d <= maxDepth; d++) {
            double acc = meanAccuracy(data, folds, d);
            scores.put(d, acc);
            if (acc > scores.get(best) + 1e-9) best = d;
        }
        return new Result(best, scores);
    }
}
