package aegisschedrl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Minimal synthetic workload generator (Poisson-like using exponential inter-arrival).
 * You can replace this with your paper-specific arrival process.
 */
public class WorkloadGenerator {
    private final Random rng;
    private int taskId = 0;

    private final double arrivalRate; // tasks per simulated second (approx)
    private final double wMin, wMax;
    private final int pMin, pMax;
    private final double dMin, dMax;

    private final double wMaxForNorm;
    private final double pMaxForNorm;
    private final double dMaxForNorm;

    public WorkloadGenerator(long seed,
                             double arrivalRate,
                             double wMin, double wMax,
                             int pMin, int pMax,
                             double dMin, double dMax) {
        this.rng = new Random(seed);
        this.arrivalRate = arrivalRate;
        this.wMin = wMin; this.wMax = wMax;
        this.pMin = pMin; this.pMax = pMax;
        this.dMin = dMin; this.dMax = dMax;

        this.wMaxForNorm = wMax;
        this.pMaxForNorm = pMax;
        this.dMaxForNorm = dMax;
    }

    public List<TaskSnapshot> sampleArrivals(double windowSeconds) {
        int k = samplePoisson(arrivalRate * windowSeconds);
        List<TaskSnapshot> tasks = new ArrayList<>(k);
        for (int i=0; i<k; i++) {
            taskId++;
            double w = uniform(wMin, wMax);
            int p = randint(pMin, pMax);
            double d = uniform(dMin, dMax);
            tasks.add(new TaskSnapshot(taskId, w, p, d, wMaxForNorm, pMaxForNorm, dMaxForNorm));
        }
        return tasks;
    }

    private double uniform(double a, double b) { return a + (b-a) * rng.nextDouble(); }
    private int randint(int a, int b) { return a + rng.nextInt(b - a + 1); }

    // Knuth Poisson sampler
    private int samplePoisson(double lambda) {
        if (lambda <= 0) return 0;
        double L = Math.exp(-lambda);
        int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= rng.nextDouble();
        } while (p > L);
        return Math.max(0, k - 1);
    }
}
