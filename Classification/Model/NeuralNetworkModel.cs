using System;
using System.Collections.Generic;
using System.IO;
using Classification.Instance;
using Classification.Parameter;
using Math;

namespace Classification.Model
{
    public abstract class NeuralNetworkModel : ValidatedModel
    {
        protected List<string> classLabels;
        protected int K, d;
        protected Vector x, y, r;

        protected abstract void CalculateOutput();

        /**
         * <summary> Constructor that sets the class labels, their sizes as K and the size of the continuous attributes as d.</summary>
         *
         * <param name="trainSet">{@link InstanceList} to use as train set.</param>
         */
        public NeuralNetworkModel(InstanceList.InstanceList trainSet)
        {
            classLabels = trainSet.GetDistinctClassLabels();
            K = classLabels.Count;
            d = trainSet.Get(0).ContinuousAttributeSize();
        }

        public NeuralNetworkModel()
        {
            
        }

        /**
         * <summary> The allocateLayerWeights method returns a new {@link Matrix} with random weights.</summary>
         *
         * <param name="row">   Number of rows.</param>
         * <param name="column">Number of columns.</param>
         * <returns>Matrix with random weights.</returns>
         */
        protected Matrix AllocateLayerWeights(int row, int column, Random random)
        {
            return new Matrix(row, column, -0.01, +0.01, random);
        }

        /**
         * <summary> The normalizeOutput method takes an input {@link Vector} o, gets the result for e^o of each element of o,
         * then sums them up. At the end, divides the each e^o by the summation.</summary>
         *
         * <param name="o">Vector to normalize.</param>
         * <returns>Normalized vector.</returns>
         */
        protected Vector NormalizeOutput(Vector o)
        {
            var sum = 0.0;
            var values = new double[o.Size()];
            for (var i = 0; i < values.Length; i++)
                sum += System.Math.Exp(o.GetValue(i));
            for (var i = 0; i < values.Length; i++)
                values[i] = System.Math.Exp(o.GetValue(i)) / sum;
            return new Vector(values);
        }

        /**
         * <summary> The createInputVector method takes an {@link Instance} as an input. It converts given Instance to the {@link java.util.Vector}
         * and insert 1.0 to the first element.</summary>
         *
         * <param name="instance">Instance to insert 1.0.</param>
         */
        protected void CreateInputVector(Instance.Instance instance)
        {
            x = instance.ToVector();
            x.Insert(0, 1.0);
        }

        /**
         * <summary> The calculateHidden method takes a {@link Vector} input and {@link Matrix} weights, It multiplies the weights
         * Matrix with given input Vector than applies the sigmoid function and returns the result.</summary>
         *
         * <param name="input">  Vector to multiply weights.</param>
         * <param name="weights"> Matrix is multiplied with input Vector.</param>
         * <param name="activationFunction"> Activation function.</param>
         * <returns>Result of sigmoid function.</returns>
         */
        protected Vector CalculateHidden(Vector input, Matrix weights, ActivationFunction activationFunction)
        {
            var z = weights.MultiplyWithVectorFromRight(input);
            switch (activationFunction)
            {
                case ActivationFunction.SIGMOID:
                    z.Sigmoid();
                    break;
                case ActivationFunction.TANH:
                    z.Tanh();
                    break;
                case ActivationFunction.RELU:
                    z.Relu();
                    break;
            }

            return z;
        }

        /**
         * <summary> The calculateOneMinusHidden method takes a {@link java.util.Vector} as input. It creates a Vector of ones and
         * returns the difference between given Vector.</summary>
         *
         * <param name="hidden">Vector to find difference.</param>
         * <returns>Returns the difference between ones Vector and input Vector.</returns>
         */
        protected Vector CalculateOneMinusHidden(Vector hidden)
        {
            var one = new Vector(hidden.Size(), 1.0);
            return one.Difference(hidden);
        }

        /**
         * <summary> The calculateForwardSingleHiddenLayer method takes two matrices W and V. First it multiplies W with x, then
         * multiplies V with the result of the previous multiplication.</summary>
         *
         * <param name="W">Matrix to multiply with x.</param>
         * <param name="V">Matrix to multiply.</param>
         * <param name="activationFunction"> Activation function.</param>
         */
        protected void CalculateForwardSingleHiddenLayer(Matrix W, Matrix V, ActivationFunction activationFunction)
        {
            var hidden = CalculateHidden(x, W, activationFunction);
            var hiddenBiased = hidden.Biased();
            y = V.MultiplyWithVectorFromRight(hiddenBiased);
        }

        /**
         * <summary> The calculateRMinusY method creates a new {@link java.util.Vector} with given Instance, then it multiplies given
         * input Vector with given weights Matrix. After normalizing the output, it return the difference between the newly created
         * Vector and normalized output.</summary>
         *
         * <param name="instance">Instance is used to get class labels.</param>
         * <param name="input">   Vector to multiply weights.</param>
         * <param name="weights"> Matrix of weights/</param>
         * <returns>Difference between newly created Vector and normalized output.</returns>
         */
        protected Vector CalculateRMinusY(Instance.Instance instance, Vector input, Matrix weights)
        {
            r = new Vector(K, classLabels.IndexOf(instance.GetClassLabel()), 1.0);
            var o = weights.MultiplyWithVectorFromRight(input);
            y = NormalizeOutput(o);
            return r.Difference(y);
        }

        /**
         * <summary> The predictWithCompositeInstance method takes an List possibleClassLabels. It returns the class label
         * which has the maximum value of y.</summary>
         *
         * <param name="possibleClassLabels">List that has the class labels.</param>
         * <returns>The class label which has the maximum value of y.</returns>
         */
        protected string PredictWithCompositeInstance(List<string> possibleClassLabels)
        {
            var predictedClass = possibleClassLabels[0];

            var maxY = double.MinValue;
            for (var i = 0; i < classLabels.Count; i++)
            {
                if (possibleClassLabels.Contains(classLabels[i]) && y.GetValue(i) > maxY)
                {
                    maxY = y.GetValue(i);
                    predictedClass = classLabels[i];
                }
            }

            return predictedClass;
        }

        /**
         * <summary> The predict method takes an {@link Instance} as an input, converts it to a Vector and calculates the {@link Matrix} y by
         * multiplying Matrix W with {@link Vector} x. Then it returns the class label which has the maximum y value.</summary>
         *
         * <param name="instance">Instance to predict.</param>
         * <returns>The class label which has the maximum y.</returns>
         */
        public override string Predict(Instance.Instance instance)
        {
            CreateInputVector(instance);

            CalculateOutput();
            if (instance is CompositeInstance compositeInstance)
            {
                return PredictWithCompositeInstance(compositeInstance.GetPossibleClassLabels());
            }

            return classLabels[y.MaxIndex()];
        }

        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            CreateInputVector(instance);

            CalculateOutput();
            var result = new Dictionary<string, double>();
            for (var i = 0; i < classLabels.Count; i++)
            {
                result[classLabels[i]] = y.GetValue(i);
            }

            return result;
        }

        protected void LoadClassLabels(StreamReader input)
        {
            var items = input.ReadLine().Split(" ");
            K = int.Parse(items[0]);
            d = int.Parse(items[1]);
            classLabels = new List<string>();
            for (var i = 0; i < K; i++)
            {
                classLabels.Add(input.ReadLine());
            }
        }

        protected ActivationFunction LoadActivationFunction(StreamReader input)
        {
            switch (input.ReadLine())
            {
                case "TANH":
                    return ActivationFunction.TANH;
                case "RELU":
                    return ActivationFunction.RELU;
                default:
                    return ActivationFunction.SIGMOID;
            }
        }
    }
}