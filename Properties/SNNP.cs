using System;

public class SNNP
{
    int input_nodes, hidden_nodes, output_nodes;
    Matrix w_ih, w_ho, bias_h, bias_o;

    public SNNP(int i, int h, int o)
    {
        input_nodes = i;
        hidden_nodes = h;
        output_nodes = o;

        w_ih = new Matrix(hidden_nodes, input_nodes);
        w_ho = new Matrix(output_nodes, hidden_nodes);

        w_ih.Randomize();
        w_ho.Randomize();

        bias_h = new Matrix(hidden_nodes, 1);
        bias_o = new Matrix(output_nodes, 1);

        bias_h.Randomize();
        bias_o.Randomize();
    }

    // For now with the sigmoid function
    public double[] Feedforward(double[] inputs)
    {
        // Convert the input array into a matrix
        Matrix input_matrix = new Matrix(inputs);

        Matrix hidden = Matrix.AddMatrices(Matrix.Multiply(w_ih, input_matrix), bias_h);
        hidden.Map(Sigmoid);

        // Aka guess
        Matrix output = Matrix.AddMatrices(Matrix.Multiply(w_ho, hidden), bias_o);
        output.Map(Sigmoid);

        return output.ToArray();
    }

    public double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

    public double DSigmoid(double x) => Sigmoid(x) * (1 - Sigmoid(x));

    // TO DO
    public double[] Train(double[] inputs, double[] targetArray, double eta = 1e-3)
    {
        #region Feedforward stuff

        // Convert the input array into a matrix
        Matrix input_matrix = new Matrix(inputs);

        Matrix hidden = Matrix.AddMatrices(Matrix.Multiply(w_ih, input_matrix), bias_h);
        hidden.Map(Sigmoid);

        Matrix guess = Matrix.AddMatrices(Matrix.Multiply(w_ho, hidden), bias_o);
        guess.Map(Sigmoid);

        #endregion

        Matrix targets = new Matrix(targetArray);
        
        // Calculate the error
        // Error = target - guess
        Matrix o_errors = Matrix.Subtract(targets, guess);

        return null;
    }
}