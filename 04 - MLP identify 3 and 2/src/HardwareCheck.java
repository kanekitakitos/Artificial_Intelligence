
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

/**
 * Utilitário para verificar se o ND4J está a utilizar a GPU (CUDA) ou a CPU.
 */
public class HardwareCheck {

    public static void main(String[] args) {
        logSystemInfo();
    }

    /**
     * Verifica e imprime o backend atual e realiza um teste de desempenho simples.
     */
    public static void logSystemInfo() {
        System.out.println("========== HARDWARE CHECK ==========");

        // 1. Verificar o Backend carregado
        String backendName = Nd4j.getBackend().getClass().getSimpleName();
        System.out.println("Backend Inicializado: " + backendName);

        boolean isGpu = backendName.toLowerCase().contains("cuda") || backendName.toLowerCase().contains("jcublas");

        if (isGpu) {
            System.out.println("✅ STATUS: GPU (NVIDIA CUDA) DETECTADA!");

            // Tenta obter informações sobre os dispositivos
            try {
                int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
                System.out.println("   -> GPUs disponíveis: " + numDevices);
                System.out.println("   -> Device ID atual: " + Nd4j.getAffinityManager().getDeviceForCurrentThread());
            } catch (Exception e) {
                System.out.println("   -> Não foi possível obter detalhes da GPU.");
            }
        } else {
            System.out.println("⚠️ STATUS: A rodar em CPU.");
            System.out.println("   -> Se tens uma GPU NVIDIA, verifica as dependências do Maven/Gradle (nd4j-cuda-platform).");
            System.out.println("   -> Verifica se o CUDA Toolkit e cuDNN estão instalados corretamente.");
        }

        // 2. Verificar o Executioner (quem processa as operações)
        OpExecutioner executioner = Nd4j.getExecutioner();
        System.out.println("Executioner: " + executioner.getClass().getSimpleName());

        // 3. Teste Rápido de Cálculo
        System.out.println("\n[Teste de Cálculo...]");
        try {
            long startTime = System.nanoTime();

            // Criação de matrizes (move dados para a GPU se disponível)
            INDArray a = Nd4j.rand(2000, 2000);
            INDArray b = Nd4j.rand(2000, 2000);

            // Multiplicação de Matrizes (operação pesada)
            a.mmul(b);

            long endTime = System.nanoTime();
            double durationMs = (endTime - startTime) / 1_000_000.0;

            System.out.println("✅ Teste concluído com sucesso.");
            System.out.println("⏱️ Tempo da operação (2000x2000 mmul): " + String.format("%.2f", durationMs) + " ms");

        } catch (Exception e) {
            System.err.println("❌ Erro ao tentar executar cálculos: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("====================================");
    }
}