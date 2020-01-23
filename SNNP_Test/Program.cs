using SNNP.MLP;
using System.IO;

namespace SNNP_Test
{
    class Program
    {
        static void Main(string[] args)
        {
            // Hidden layers
            int[] h_n = { 2, 3 };

            // XOR dataset
            double[,] dataset = { { 1, 1, 0 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 1, 1 } };

            // Architecture
            var mlp = new MLP(2, h_n, 1, Activation.LeakyReLU, Activation.DLeakyReLU);

            // Training
            var r = mlp.Backpropagation(dataset);

            StreamWriter sw = new StreamWriter("Test.txt");

            foreach (double d in r)
            {
                sw.WriteLine(d);
            }

            sw.Close();
        }
    }
}