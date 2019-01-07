using System;
using System.Collections.Generic;

public class Matrix
{
    public int rows, columns;
    double[,] data;
    private Matrix matrix;

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
        rows = 1;
        columns = inputs.Length;
        data = new double[rows, columns];

        for (int i = 0; i < columns; i++)
            data[0, i] = inputs[i];
    }

    public double[] ToArray()
    {
        List<double> r = new List<double>();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                r.Add(data[i, j]);
        return r.ToArray();
    }

    public void AddColumns(int nColumns, bool random = true)
    {
        int oldNColumns = columns;
        columns += nColumns;

        double[,] new_data = new double[rows, columns];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < oldNColumns; j++)
                new_data[i, j] = data[i, j];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < nColumns; j++)
                new_data[i, oldNColumns + j] = (random)? Utility.NextDouble(-.5F, .5F) : 1;

        data = new_data;
    }

    public void Randomize()
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                data[i, j] = Utility.NextDouble(-.5F, .5F);
    }

    public void T()
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
    public Matrix Map(Func<double, double> func)
    {
        Matrix r = this;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                r.data[i, j] = func(data[i, j]);
        return r;
    }

    public static Matrix T(Matrix a)
    {
        Matrix r = new Matrix(a.columns, a.rows);

        for (int i = 0; i < a.rows; i++)
            for (int j = 0; j < a.columns; j++)
                r.data[j, i] = a.data[i, j];
        return r;
    }

    public double SumSquared()
    {
        double sum = 0;
        for (int i = 0; i < columns; i++)
            sum += data[0, i] * data[0, i];
        return sum;
    }

    public static Matrix Add(Matrix a, double n)
    {
        Matrix r = new Matrix(a.rows, a.columns);

        for (int i = 0; i < a.rows; i++)
            for (int j = 0; j < a.columns; j++)
                r.data[i, j] = a.data[i, j] + n;
        return r;
    }

    public Matrix SubMatrix(int[] rowsRange, int[] columnsRange)
    {
        int newRows = rowsRange[1] - rowsRange[0];
        int newColumns = columnsRange[1] - columnsRange[0];
        Matrix r = new Matrix(newRows, newColumns);

        for (int i = 0; i < newRows; i++)
            for (int j = 0; j < newColumns; j++)
                r.data[i, j] = data[rowsRange[0] - 1 + i, columnsRange[0] - 1 + j];
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

    public static Matrix MultiplyT(Matrix a, Matrix b)
    {
        Matrix r = new Matrix(a.rows, a.columns);

        for (int i = 0; i < r.rows; i++)
            for (int j = 0; j < r.columns; j++)
                r.data[i, j] = a.data[i, j] * b.data[i, j];

        return r;
    }
}