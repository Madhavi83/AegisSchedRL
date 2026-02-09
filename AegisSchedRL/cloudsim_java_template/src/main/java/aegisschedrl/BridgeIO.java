package aegisschedrl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ArrayNode;

import java.io.IOException;
import java.nio.file.*;
import java.time.Instant;
import java.util.List;
import java.util.Map;

public class BridgeIO {
    private final Path dir;
    private final ObjectMapper mapper = new ObjectMapper();

    public BridgeIO(String bridgeDir) {
        this.dir = Paths.get(bridgeDir);
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create bridge dir: " + dir, e);
        }
    }

    public Path path(String name) { return dir.resolve(name); }

    public void writeNodes(List<NodeSnapshot> nodes) throws IOException {
        ObjectNode root = mapper.createObjectNode();
        ArrayNode arr = root.putArray("nodes");
        for (NodeSnapshot n : nodes) {
            ObjectNode o = arr.addObject();
            o.put("node_id", n.nodeId());
            o.put("node_type", n.nodeType());
            o.put("cpu_avail", n.cpuAvail());
            o.put("cpu_max", n.cpuMax());
            o.put("mem_avail", n.memAvail());
            o.put("mem_max", n.memMax());
            if (n.energyAvail() == null) o.putNull("energy_avail"); else o.put("energy_avail", n.energyAvail());
            if (n.energyMax() == null) o.putNull("energy_max"); else o.put("energy_max", n.energyMax());
            o.put("queue_len", n.queueLen());
            o.put("queue_max", n.queueMax());
            o.put("lat_ms", n.latMs());
            o.put("bw_mbps", n.bwMbps());
            o.put("bw_max", n.bwMax());
        }
        Files.writeString(path("nodes.json"), mapper.writerWithDefaultPrettyPrinter().writeValueAsString(root));
    }

    public void writeTask(TaskSnapshot t) throws IOException {
        ObjectNode o = mapper.createObjectNode();
        o.put("task_id", t.taskId());
        o.put("workload", t.workload());
        o.put("priority", t.priority());
        o.put("deadline", t.deadline());
        o.put("workload_max", t.workloadMax());
        o.put("priority_max", t.priorityMax());
        o.put("deadline_max", t.deadlineMax());
        Files.writeString(path("task.json"), mapper.writerWithDefaultPrettyPrinter().writeValueAsString(o));
    }

    public void markStepReady() throws IOException {
        Files.writeString(path("step.done"), "");
    }

    public Action readActionBlocking(long timeoutMillis, long pollMillis) throws IOException, InterruptedException {
        long start = System.currentTimeMillis();
        Path actionDone = path("action.done");
        Path actionJson = path("action.json");
        while (true) {
            if (Files.exists(actionDone) && Files.exists(actionJson)) {
                var tree = mapper.readTree(Files.readString(actionJson));
                int actionIndex = tree.get("action_index").asInt();
                int nodeId = tree.get("node_id").asInt();
                String nodeType = tree.get("node_type").asText();
                // consume marker
                try { Files.deleteIfExists(actionDone); } catch (IOException ignore) {}
                return new Action(actionIndex, nodeId, nodeType);
            }
            if (System.currentTimeMillis() - start > timeoutMillis) {
                throw new IOException("Timed out waiting for action files in " + dir);
            }
            Thread.sleep(pollMillis);
        }
    }

    public void writeOutcome(double delay, double energy, boolean slaSatisfied) throws IOException {
        ObjectNode o = mapper.createObjectNode();
        o.put("delay", delay);
        o.put("energy", energy);
        o.put("sla_satisfied", slaSatisfied);
        Files.writeString(path("outcome.json"), mapper.writerWithDefaultPrettyPrinter().writeValueAsString(o));
        Files.writeString(path("outcome.done"), "");
    }

    public record Action(int actionIndex, int nodeId, String nodeType) {}
}
