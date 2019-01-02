using System;
using System.Collections.Generic;

public class Matrix
{
    int rows, columns;
    double[,] data;

    // Constructor 1
    public Matrix(int rows_, int columns_)
    {
        rows = rows_;
        columns = columns_;
        data = new double[rows, columns];
    }

    // Constructor 2
    public Matrix(double[] inputs)
    {
        rows = inputs.Length;
        columns = 1;
        data = new double[rows, columns];

        for (int i = 0; i < rows; i++)
            data[i, 0] = inputs[i];
    }

    public double[] ToArray()
    {
        List<double> r = new List<double>();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                r.Add(data[i, j]);
        return r.ToArray();
    }

    public void Randomize()
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                data[i, j] = Utility.NextDouble(-.5F, .5F);
    }

    public void Transpose()
    {
        Matrix r = new Matrix(columns, rows);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                r.data[j, i] = data[i, j];
        rows = r.rows;
        columns = r.columns;
        data = r.data;
    }

    // Apply a random function that receives a double and returns a double at each spot in the matrix
    public void Map(Func<double, double> func)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                data[i, j] = func(data[i, j]);
    }

    public static Matrix Transpose(Matrix a)
    {
        Matrix r = new Matrix(a.columns, a.rows);

        for (int i = 0; i < a.rows; i++)
            for (int j = 0; j < a.columns; j++)
                r.data[j, i] = a.data[i, j];
        return r;
    }

    public static Matrix Add(Matrix a, double n)
    {
        Matrix r = new Matrix(a.rows, a.columns);

        for (int i = 0; i < a.rows; i++)
            for (int j = 0; j < a.columns; j++)
                r.data[i, j] = a.data[i, j] + n;
        return r;
    }

    // Returns a new matrix a-b
    public static Matrix Subtract(Matrix a, Matrix b)
    {
        Matrix r = new Matrix(a.rows, a.columns);

        for (int i = 0; i < a.rows; i++)
            for (int j = 0; j < a.columns; j++)
                r.data[i, j] = a.data[i, j] - b.data[i, j];
        return r;
    }

    public static Matrix Scalar(Matrix a, double n)
    {
        Matrix r = new Matrix(a.rows, a.columns);

        for (int i = 0; i < a.rows; i++)
            for (int j = 0; j < a.columns; j++)
                r.data[i, j] = a.data[i, j] * n;
        return r;
    }

    public static Matrix AddMatrices(Matrix a, Matrix b)
    {
        // Matrix 'a' must be at least the same size of 'b'
        Matrix r = new Matrix(a.rows, a.columns);

        for (int i = 0; i < b.rows; i++)
            for (int j = 0; j < b.columns; j++)
                r.data[i, j] += a.data[i, j] + b.data[i, j];
        return r;
    }

    public static Matrix Multiply(Matrix a, Matrix b)
    {
        if (a.columns != b.rows) return null;

        Matrix r = new Matrix(a.rows, b.columns);

        for (int i = 0; i < r.rows; i++)
            for (int j = 0; j < r.columns; j++)
            {
                double sum = 0;
                for (int k = 0; k < a.columns; k++) // a.columns or b.rows
                    sum += a.data[i, k] * b.data[k, j];
                r.data[i, j] = sum;
            }
        return r;
    }
}