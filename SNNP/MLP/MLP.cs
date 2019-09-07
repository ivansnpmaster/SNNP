using System;
using System.Collections.Generic;

namespace SNNP.MLP
{
    public class MLP
    {
        public int i_n, o_n;

        private int[] h_n;

        public Matrix[] w;
        public Matrix[] b;

        private Func<double, double> a_f;
        private Func<double, double> da_f;

        public MLP(int input_nodes, int[] hidden_nodes, int output_nodes, Func<double, double> activation_function, Func<double, double> d_activation_function)
        {
            i_n = input_nodes;
            h_n = hidden_nodes;
            o_n = output_nodes;

            w = new Matrix[hidden_nodes.Length + 1];
            b = new Matrix[hidden_nodes.Length + 1];

            w[0] = new Matrix(h_n[0], i_n);
            b[0] = new Matrix(h_n[0], 1);

            //for (int i = 1; i < w.Length - 1; i++)
            //{
            //    w[i] = new Matrix(h_n[i], h_n[i - 1], true);
            //    b[i] = new Matrix(h_n[i], 1, true);
            //}

            w[w.Length - 1] = new Matrix(o_n, h_n[h_n.Length - 1]);
            b[b.Length - 1] = new Matrix(o_n, 1);

            a_f = activation_function;
            da_f = d_activation_function;
        }

        public Tuple<Matrix[], Matrix[], Matrix> Feedforward(double[] input_nodes)
        {
            Matrix inputs = new Matrix(input_nodes);

            Matrix[] net = new Matrix[h_n.Length + 1];
            Matrix[] fnet = new Matrix[h_n.Length + 1];

            net[0] = w[0] * inputs + b[0];
            fnet[0] = Matrix.Map(net[0], a_f);

            //for (int i = 1; i < net.Length; i++)
            //{
            //    net[i] = w[i] * fnet[i - 1] + b[i];
            //    fnet[i] = Matrix.Map(net[i], a_f);
            //}

            net[1] = w[1] * fnet[0] + b[1];
            fnet[1] = Matrix.Map(net[1], a_f);

            return new Tuple<Matrix[], Matrix[], Matrix>(net, fnet, inputs);
        }

        public List<double> Backpropagation(double[,] dataset, double eta = 0.1, double threshold = 1e-3)
        {
            List<double> ret = new List<double>();

            double rows = dataset.GetLength(0);

            const int number_iter = 100000;
            int counter = 0;

            double squaredError = 2 * threshold;

            while (counter < number_iter && squaredError > threshold)
            {
                squaredError = 0;
                counter++;

                for (int i = 0; i < rows; i++)
                {
                    double[] xp = new double[i_n];

                    for (int j = 0; j < xp.Length; j++)
                        xp[j] = dataset[i, j];

                    double[] yp = new double[dataset.GetLength(1) - i_n]; // number of outputs expected

                    for (int j = 0; j < yp.Length; j++)
                        yp[j] = dataset[i, i_n + j];

                    Matrix Yp = new Matrix(yp);

                    var ff = Feedforward(xp);

                    Matrix[] net = ff.Item1;
                    Matrix[] fnet = ff.Item2;
                    Matrix i_m = ff.Item3;

                    Matrix Op = fnet[fnet.Length - 1];

                    Matrix error = Op - Yp;

                    squaredError += error.SquaredSum();

                    // Backpropagation

                    // % hadamard product, * matrix multiplication

                    Matrix error_o = (-2 * error) % Matrix.Map(fnet[fnet.Length - 1], da_f);

                    Matrix error_h = (Matrix.T(w[w.Length - 1]) * error_o) % Matrix.Map(fnet[fnet.Length - 2], da_f);

                    Matrix gradient_wo = error_o * Matrix.T(fnet[fnet.Length - 2]);
                    Matrix gradient_bo = error_o;

                    Matrix gradient_wh = error_h * Matrix.T(i_m);
                    Matrix gradient_bh = error_h;

                    w[1] = w[1] - (eta * gradient_wo);
                    b[1] = b[1] - (eta * gradient_bo);

                    w[0] = w[0] - (eta * gradient_wh);
                    b[0] = b[0] - (eta * gradient_bh);
                }

                squaredError /= rows;

                Console.WriteLine(squaredError);

                ret.Add(squaredError);
            }

            return ret;
        }
    }
}
