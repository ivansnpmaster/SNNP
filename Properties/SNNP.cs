using System;
using System.Collections.Generic;

public class SNNP
{
    int input_nodes, hidden_nodes, output_nodes;
    Matrix hidden, output;

    // Put F and df_dnet as parameters
    public SNNP(int i, int h, int o)
    {
        input_nodes = i;
        hidden_nodes = h;
        output_nodes = o;

        hidden = new Matrix(hidden_nodes, input_nodes + 1);
        output = new Matrix(output_nodes, hidden_nodes + 1);

        hidden.Randomize();
        output.Randomize();
    }

    // For now with the sigmoid function
    public List<Matrix> FeedForward(double[] inputs)
    {
        // Convert the input array into a matrix and add one column for the bias
        Matrix input_matrix = new Matrix(inputs);
        input_matrix.AddColumns(1);

        // Hidden layer
        Matrix net_h_p = Matrix.Multiply(hidden, input_matrix);
        Matrix f_net_h_p = net_h_p.Map(F);

        // Output layer
        f_net_h_p.AddColumns(1, false); // For theta
        Matrix net_o_p = Matrix.Multiply(output, f_net_h_p);
        Matrix f_net_o_p = net_o_p.Map(F);

        // 0 -> net_h_p
        // 1 -> net_o_p
        // 2 - f_net_h_p
        // 3 -> f_net_o_p
        List<Matrix> r = new List<Matrix>() { net_h_p, net_o_p, f_net_h_p, f_net_o_p };

        return r;
    }

    public double F(double net) => 1 / (1 + Math.Exp(-net));

    public double dF_dnet(double f_net) => f_net * (1 - f_net);

    // Need testing
    public void Backpropagation(double[] inputs, double[,] dataset, double eta = .1F, double threshold = 1e-3)
    {
        double squaredError = 2 * threshold;
        int counter = 0;

        while (squaredError > threshold)
        {
            squaredError = 0;

            for (int i = 0; i < dataset.GetLength(0); i++)
            {
                double[] xp = new double[inputs.Length];

                for (int j = 0; j < xp.Length; j++)
                    xp[j] = dataset[i, j];

                double[] yp = new double[dataset.GetLength(1) - inputs.Length]; // number of outputs expected

                for (int k = 0; k < yp.Length; k++)
                    yp[k] = dataset[i, inputs.Length + k];

                Matrix Yp = new Matrix(yp);
                List<Matrix> results = FeedForward(xp);
                Matrix Op = results[3]; // f_net_o_p
                Matrix error = Matrix.Subtract(Yp, Op);

                squaredError += error.SumSquared();

                // Training output proccess
                // delta_o = (Yp - Op) * f_o_p'(net_o_p)
                // w(t+1) = w(t) - eta * dE2_dw
                // w(t+1) = w(t) - eta * delta_o * i_pj

                Matrix delta_o_p = Matrix.Multiply(error, Op.Map(dF_dnet));

                // Training hidden proccess
                // delta_h = f_h_p'(net_h_p) * sum(delta_o * w_o_kj)
                // w(t+1) = w(t) - eta * delta_h * xp

                Matrix f_net_h_p = results[2]; // Declaring here because I'm changing results[2] down below
                Matrix w_o_kj = output.SubMatrix(new int[] { 1, output.rows }, new int[] { 1, hidden.columns }); // From row index 0 to output.rows - 1, same for columns
                Matrix delta_h_p = Matrix.MultiplyT(results[2].Map(dF_dnet), Matrix.Multiply(delta_o_p, w_o_kj));

                // Actual training

                f_net_h_p.AddColumns(1, false); // For theta
                output = Matrix.AddMatrices(output, Matrix.Scalar(Matrix.Multiply(delta_o_p, f_net_h_p), eta));

                Matrix Xp = new Matrix(xp);
                Xp.AddColumns(1, false); // For theta
                hidden = Matrix.AddMatrices(hidden, Matrix.Scalar(Matrix.Multiply(Matrix.T(delta_h_p), Xp), eta));
            }

            squaredError /= Math.Max(1, dataset.Length);
            counter++;
        }
    }
}