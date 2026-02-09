package aegisschedrl;

public record NodeSnapshot(
        int nodeId,
        String nodeType,
        double cpuAvail,
        double cpuMax,
        double memAvail,
        double memMax,
        Double energyAvail,
        Double energyMax,
        double queueLen,
        double queueMax,
        double latMs,
        double bwMbps,
        double bwMax
) {}

public record TaskSnapshot(
        int taskId,
        double workload,
        double priority,
        double deadline,
        double workloadMax,
        double priorityMax,
        double deadlineMax
) {}
