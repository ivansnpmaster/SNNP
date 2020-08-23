using System;

using SNNP;
using SNNP.MLP;
using SNNP.kMeans;

namespace SNNP_Test
{
    class Program
    {
        static void Main(string[] _)
        {
            // Multilayer Perceptron example
            //MLPExample();

            // KMeans example
            KMeansExample();

            // Eigenvalue Power Method example
            //EPMExample();

            Console.ReadLine();
        }

        private static void KMeansExample()
        {
            double[,] data ={
                {3.0, 1.5},
                {2.0, 1.0},
                {4.0, 1.5},
                {3.0, 1.0},
                {3.5, 0.5},
                {2.0, 0.5},
                {5.5, 1.0},
                {1.0, 1.0},
            };

            var kmeans = new KMeans(data, 2, 10, true);

            Console.WriteLine("Done");
        }

        private static void MLPExample()
        {
            // Hidden layers
            int[] h_n = { 2 };

            // XOR dataset
            double[,] dataset = { { 1, 1, 0 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 1, 1 } };

            // Architecture
            var mlp = new MLP(2, h_n, 1, Activation.BentIdentity, Activation.DBentIdentity);

            // Optional -> fit into [0, 1] range
            //dataset = Utility.NormalizeData(dataset);
            dataset = Mathf.Standardize(dataset);

            // Training
            var r = mlp.Backpropagation_Momentum(dataset, threshold: 1e-15);

            // Saving into a file the mean squared error over the epochs
            using (var sw = new System.IO.StreamWriter("MLPExample.txt"))
                foreach (double d in r)
                    sw.WriteLine(d);

            //Utility.Save<MLP>("C:\\Users\\ivan.soares\\Desktop\\mlp", mlp);

            //var o = Utility.Load<MLP>("C:\\Users\\ivan.soares\\Desktop\\mlp.bin");
            //Console.WriteLine(o.ToString());

            Console.WriteLine("Done");
            Console.ReadLine();
        }

        private static void EPMExample()
        {
            double[,] d = new double[,]
            {
                {7, 9},
                {9, 7}
            };

            var epm = Mathf.EPM(d);

            Console.WriteLine($"Eigenvalue: {epm.Item1}");
            Console.WriteLine($"Eigenvector: [{epm.Item2.data[0, 0]}, {epm.Item2.data[1, 0]}]");

            Console.ReadLine();
        }
    }
}