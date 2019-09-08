using System;
using System.IO;
using SNNP.MLP;

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

            //double niter = 100;

            //for (int i = 0; i < niter; i++)
            //    for (int j = 0; j < niter; j++)
            //    {
            //        double ii = i / niter;
            //        double jj = j / niter;
            //        double[] re = { ii, jj };

            //        var fnet = mlp.Feedforward(re).Item2;

            //        sw.WriteLine(string.Format("{0} {1} {2}", ii, jj, fnet[fnet.Length - 1].data[0, 0]));
            //    }

            foreach (double d in r)
            {
                sw.WriteLine(d);
            }

            sw.Close();
        }
    }
}