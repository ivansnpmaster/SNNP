using System;
using System.Collections.Generic;

namespace SNNP.MLP
{
    [Serializable()]
    public class MLP
    {
        public int i_n, o_n;

        private readonly int[] h_n;

        public Matrix[] w;
        public Matrix[] b;

        private readonly Func<double, double> a_f;
        private readonly Func<double, double> da_f;

        // To store the momentum from the last training step
        private Matrix[] wLastGradients;
        private Matrix[] bLastGradients;

        /// <summary>
        /// Creates a Multilayer Perceptron with support to multiple hidden layers.
        /// </summary>
        /// <param name="input_nodes">Number of inputs that the model will receive.</param>
        /// <param name="hidden_nodes">Organization of the hidden layers.</param>
        /// <param name="output_nodes">Number of outputs that the model will produce.</param>
        /// <param name="activation_function">Activation function to produce an output.</param>
        /// <param name="d_activation_function">The derivative of the activation function to train the model.</param>
        public MLP(int input_nodes, int[] hidden_nodes, int output_nodes, Func<double, double> activation_function, Func<double, double> d_activation_function)
        {
            i_n = input_nodes;
            h_n = hidden_nodes;
            o_n = output_nodes;

            w = new Matrix[hidden_nodes.Length + 1];
            b = new Matrix[hidden_nodes.Length + 1];

            w[0] = new Matrix(h_n[0], i_n);
            b[0] = new Matrix(h_n[0], 1);

            for (int i = 1; i < w.Length - 1; i++)
            {
                w[i] = new Matrix(h_n[i], h_n[i - 1]);
                b[i] = new Matrix(h_n[i], 1);
            }

            w[w.Length - 1] = new Matrix(o_n, h_n[h_n.Length - 1]);
            b[b.Length - 1] = new Matrix(o_n, 1);

            a_f = activation_function;
            da_f = d_activation_function;
        }

        /// <summary>
        /// Feedforward the given inputs.
        /// </summary>
        /// <param name="input_nodes">Input data to predict.</param>
        /// <returns>A tuple with net (as Matrix[]), fnet (as Matrix[]) and input (as Matrix).</returns>
        public Tuple<Matrix[], Matrix[], Matrix> Feedforward(double[] input_nodes)
        {
            Matrix input = new Matrix(input_nodes);

            Matrix[] net = new Matrix[h_n.Length + 1];
            Matrix[] fnet = new Matrix[h_n.Length + 1];

            net[0] = w[0] * input + b[0];
            fnet[0] = Matrix.Map(net[0], a_f);

            for (int i = 1; i < net.Length; i++)
            {
                net[i] = w[i] * fnet[i - 1] + b[i];
                fnet[i] = Matrix.Map(net[i], a_f);
            }

            return new Tuple<Matrix[], Matrix[], Matrix>(net, fnet, input);
        }

        /// <summary>
        /// Trains the model based on a dataset.
        /// </summary>
        /// <param name="dataset">Dataset to train the model. It must match the model construction parameters in order to predict multiple values.</param>
        /// <param name="eta">To reduce the intensity of the gradient.</param>
        /// <param name="threshold">Threshold to stop the training process.</param>
        /// <param name="iterations">Maximum number of iterations to stop the training process.</param>
        /// <returns>A list with the mean squared error over each epoch.</returns>
        public List<double> Backpropagation(double[,] dataset, double eta = 0.01, double threshold = 1e-3, int iterations = 50000)
        {
            List<double> ret = new List<double>();

            double rows = dataset.GetLength(0);
            int cols = dataset.GetLength(1);
            int counter = 0;

            double squaredError = 2 * threshold;

            while (counter < iterations && squaredError > threshold)
            {
                squaredError = 0;
                counter++;

                for (int i = 0; i < rows; i++)
                {
                    double[] xp = new double[i_n];

                    for (int j = 0; j < xp.Length; j++)
                        xp[j] = dataset[i, j];

                    double[] yp = new double[cols - i_n]; // number of outputs expected

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

                    Matrix[] errors = new Matrix[h_n.Length + 1];

                    Matrix[] w_gradients = new Matrix[h_n.Length + 1];
                    Matrix[] b_gradients = new Matrix[h_n.Length + 1];

                    // Output error
                    errors[errors.Length - 1] = (2 * error) % Matrix.Map(net[net.Length - 1], da_f);
                    w_gradients[w_gradients.Length - 1] = errors[errors.Length - 1] * Matrix.T(fnet[fnet.Length - 2]);
                    b_gradients[b_gradients.Length - 1] = errors[errors.Length - 1];

                    // First hidden layer must have an operation with the inputs, that's why j > 0
                    for (int j = errors.Length - 2; j > 0; j--)
                    {
                        errors[j] = (Matrix.T(w[j + 1]) * errors[j + 1]) % Matrix.Map(net[j], da_f);
                        w_gradients[j] = errors[j] * Matrix.T(fnet[j - 1]);
                        b_gradients[j] = errors[j];
                    }

                    errors[0] = (Matrix.T(w[1]) * errors[1]) % Matrix.Map(net[0], da_f);
                    w_gradients[0] = errors[0] * Matrix.T(i_m);
                    b_gradients[0] = errors[0];

                    for (int j = w.Length - 1; j > -1; j--)
                    {
                        w[j] -= eta * w_gradients[j];
                        b[j] -= eta * b_gradients[j];
                    }
                }

                squaredError /= rows;

                Console.WriteLine(string.Format("{0} - {1}", counter, squaredError));

                ret.Add(squaredError);
            }

            return ret;
        }

        /// <summary>
        /// Trains the model based on a dataset using the momentum concept.
        /// </summary>
        /// <param name="dataset">Dataset to train the model. It must match the model construction parameters in order to predict multiple values.</param>
        /// <param name="eta">To reduce the intensity of the gradient.</param>
        /// <param name="threshold">Threshold to stop the training process.</param>
        /// <param name="iterations">Maximum number of iterations to stop the training process.</param>
        /// <param name="k">To reduce the intensity of the momentum.</param>
        /// <returns></returns>
        public List<double> Backpropagation_Momentum(double[,] dataset, double eta = .01, double threshold = 1e-3, int iterations = 50000, double k = .9)
        {
            List<double> ret = new List<double>();

            // To perform a division later on
            double rows = dataset.GetLength(0);
            int cols = dataset.GetLength(1);
            int counter = 0;

            #region Momentum initialization

            wLastGradients = new Matrix[h_n.Length + 1];
            bLastGradients = new Matrix[h_n.Length + 1];

            wLastGradients[wLastGradients.Length - 1] = new Matrix(o_n, h_n[h_n.Length - 1], true);
            bLastGradients[bLastGradients.Length - 1] = new Matrix(o_n, 1, true);

            for (int i = wLastGradients.Length - 2; i > 0; i--)
            {
                wLastGradients[i] = new Matrix(h_n[i], h_n[i - 1], true);
                bLastGradients[i] = new Matrix(h_n[i], 1, true);
            }

            wLastGradients[0] = new Matrix(h_n[0], i_n, true);
            bLastGradients[0] = new Matrix(h_n[0], 1, true);

            #endregion

            double squaredError = 2 * threshold;

            while (counter < iterations && squaredError > threshold)
            {
                squaredError = 0;
                counter++;

                for (int i = 0; i < rows; i++)
                {
                    double[] xp = new double[i_n];

                    for (int j = 0; j < xp.Length; j++)
                        xp[j] = dataset[i, j];

                    double[] yp = new double[cols - i_n]; // number of outputs expected

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

                    Matrix[] errors = new Matrix[h_n.Length + 1];

                    Matrix[] w_gradients = new Matrix[h_n.Length + 1];
                    Matrix[] b_gradients = new Matrix[h_n.Length + 1];

                    // Output error
                    errors[errors.Length - 1] = (2 * error) % Matrix.Map(net[net.Length - 1], da_f);
                    w_gradients[w_gradients.Length - 1] = errors[errors.Length - 1] * Matrix.T(fnet[fnet.Length - 2]);
                    b_gradients[b_gradients.Length - 1] = errors[errors.Length - 1];

                    // First hidden layer must have an operation with the inputs, that's why j > 0
                    for (int j = errors.Length - 2; j > 0; j--)
                    {
                        errors[j] = (Matrix.T(w[j + 1]) * errors[j + 1]) % Matrix.Map(net[j], da_f);
                        w_gradients[j] = errors[j] * Matrix.T(fnet[j - 1]);
                        b_gradients[j] = errors[j];
                    }

                    errors[0] = (Matrix.T(w[1]) * errors[1]) % Matrix.Map(net[0], da_f);
                    w_gradients[0] = errors[0] * Matrix.T(i_m);
                    b_gradients[0] = errors[0];

                    for (int j = w.Length - 1; j > -1; j--)
                    {
                        w[j] -= eta * w_gradients[j] + k * wLastGradients[j];
                        b[j] -= eta * b_gradients[j] + k * bLastGradients[j];

                        wLastGradients[j] = k * w_gradients[j];
                        bLastGradients[j] = k * bLastGradients[j];
                    }
                }

                squaredError /= rows;

                Console.WriteLine(string.Format("{0} - {1}", counter, squaredError));

                ret.Add(squaredError);
            }

            return ret;
        }
    }
}