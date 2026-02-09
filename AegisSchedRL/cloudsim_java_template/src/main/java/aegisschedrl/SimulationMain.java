package aegisschedrl;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudsimplus.builders.tables.CloudletsTableBuilder;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Minimal CloudSim Plus template that:
 *  1) Creates edge+cloud VMs (as targets)
 *  2) Generates tasks (cloudlets)
 *  3) For each task: writes nodes/task snapshots to bridge, waits action.json,
 *     assigns cloudlet to chosen VM, runs the simulation step, and writes outcome.
 *
 * NOTE:
 *  - This is a template to demonstrate the bridge protocol.
 *  - You should adapt node snapshots and metrics to match your paper (CPU/mem/queue/lat/bw).
 */
public class SimulationMain {

    public static void main(String[] args) throws Exception {
        String bridgeDir = "bridge";
        BridgeIO io = new BridgeIO(bridgeDir);

        CloudSim sim = new CloudSim();

        // --- Create hosts and datacenters (edge + cloud) ---
        Datacenter edgeDC = createDatacenter(sim, 2, 4, 1000);   // 2 hosts, 4 PEs, 1000 MIPS each
        Datacenter cloudDC = createDatacenter(sim, 1, 16, 5000); // 1 host, 16 PEs, 5000 MIPS each

        // --- Create target VMs (represent edge nodes + cloud servers) ---
        List<Vm> vms = new ArrayList<>();
        // Edge VMs
        vms.add(createVm(1, 2, 1000, 2048));
        vms.add(createVm(2, 2, 1000, 2048));
        // Cloud VM(s)
        vms.add(createVm(3, 8, 5000, 16384));

        // Place VMs: edge VMs into edgeDC hosts, cloud VMs into cloudDC hosts
        placeVms(edgeDC, vms.subList(0,2));
        placeVms(cloudDC, vms.subList(2,3));

        // Workload generator
        WorkloadGenerator wg = new WorkloadGenerator(
                7L,
                2.0,           // arrival rate (approx per window)
                50, 500,       // workload range
                1, 5,          // priority range
                5, 50          // deadline range
        );

        // Process a fixed number of decision steps
        int decisionSteps = 200;
        double windowSeconds = 1.0;

        for (int step = 0; step < decisionSteps; step++) {
            var arrivals = wg.sampleArrivals(windowSeconds);
            if (arrivals.isEmpty()) {
                continue;
            }

            for (TaskSnapshot t : arrivals) {
                // Build node snapshots (you should compute real cpu_avail, queues, etc)
                List<NodeSnapshot> nodes = buildNodeSnapshots(vms);

                // Write snapshots for Python
                io.writeNodes(nodes);
                io.writeTask(t);
                io.markStepReady();

                // Wait for Python safe action
                var action = io.readActionBlocking(30_000, 50);

                // Create a Cloudlet for this task and submit to chosen VM
                Vm targetVm = pickVmByNodeId(vms, action.nodeId());
                Cloudlet cl = createCloudletFromTask(t);
                cl.setVm(targetVm);

                // Run a small simulation slice by starting and stopping
                // (Simpler: run all at once with a broker; for template we keep it minimal)
                sim.start();

                // Collect outcome metrics (finish time as delay proxy; energy not modeled here)
                double delay = cl.getFinishTime() - cl.getSubmissionDelay();
                double energy = 0.0; // replace with energy model if available
                boolean sla = delay <= t.deadline();

                io.writeOutcome(delay, energy, sla);
            }
        }
    }

    private static Datacenter createDatacenter(CloudSim sim, int hosts, int pesPerHost, long mips) {
        List<Host> hostList = new ArrayList<>();
        for (int i=0; i<hosts; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int p=0; p<pesPerHost; p++) peList.add(new PeSimple(mips));
            Host h = new HostSimple(8192, 100_000, 1_000_000, peList);
            h.setVmScheduler(new VmSchedulerTimeShared());
            hostList.add(h);
        }
        return new DatacenterSimple(sim, hostList);
    }

    private static Vm createVm(long id, int pes, long mips, long ramMb) {
        Vm vm = new VmSimple(id, mips, pes);
        vm.setRam(ramMb).setBw(10_000).setSize(10_000);
        vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
        return vm;
    }

    private static void placeVms(Datacenter dc, List<Vm> vms) {
        // CloudSim Plus creates placement through brokers in normal usage.
        // For this template, we rely on default mapping by setting VM datacenter at runtime via cloudlets.
        // In your final code, use a DatacenterBrokerSimple and submit VMs/Cloudlets properly.
        // Keeping the template minimal and focused on the bridge.
    }

    private static Vm pickVmByNodeId(List<Vm> vms, int nodeId) {
        return vms.stream()
                .filter(v -> (int)v.getId() == nodeId)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("No VM for node_id=" + nodeId));
    }

    private static Cloudlet createCloudletFromTask(TaskSnapshot t) {
        long length = (long) Math.max(1, t.workload() * 1000); // scale workload -> MI
        int pes = 1;
        var utilization = new UtilizationModelDynamic(0.7);
        Cloudlet cl = new CloudletSimple(length, pes, utilization);
        cl.setSubmissionDelay(0);
        return cl;
    }

    private static List<NodeSnapshot> buildNodeSnapshots(List<Vm> vms) {
        List<NodeSnapshot> nodes = new ArrayList<>();
        for (Vm vm : vms) {
            // Approximate availability (in real implementation, compute remaining capacity)
            double cpuMax = vm.getTotalMipsCapacity();
            double cpuAvail = cpuMax; // template placeholder
            double memMax = vm.getRam().getCapacity();
            double memAvail = memMax; // placeholder
            double queueLen = 0;      // placeholder
            double queueMax = 50;

            String type = (vm.getId() <= 2) ? "edge" : "cloud";
            double lat = (type.equals("edge") ? 20 : 120);
            double bw = (type.equals("edge") ? 50 : 200);
            double bwMax = (type.equals("edge") ? 100 : 500);

            nodes.add(new NodeSnapshot(
                    (int) vm.getId(), type,
                    cpuAvail, cpuMax,
                    memAvail, memMax,
                    null, null,
                    queueLen, queueMax,
                    lat, bw, bwMax
            ));
        }
        // Keep stable ordering: edge first then cloud
        nodes.sort(Comparator.comparing(NodeSnapshot::nodeType).thenComparing(NodeSnapshot::nodeId));
        return nodes;
    }
}
