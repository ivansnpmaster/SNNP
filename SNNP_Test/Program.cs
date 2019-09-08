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

            double[,] dataset = { { 1, 1, 0 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 1, 1 } };

            var mlp = new MLP(2, h_n, 1, Activation.LeakyReLU, Activation.DLeakyReLU);

            var r = mlp.Backpropagation(dataset, iterations: 1000000);

            StreamWriter sw = new StreamWriter("Test.txt");

            //foreach (double d in r)
            //{
            //    sw.WriteLine(d);
            //}

            double niter = 100;

            for (int i = 0; i < niter; i++)
                for (int j = 0; j < niter; j++)
                {
                    double ii = i / niter;
                    double jj = j / niter;
                    double[] re = { ii, jj };

                    var fnet = mlp.Feedforward(re).Item2;

                    sw.WriteLine(string.Format("{0} {1} {2}", ii, jj, fnet[fnet.Length - 1].data[0, 0]));
                }

            sw.Close();
        }
    }
}