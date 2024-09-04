using System.Collections.Generic;
using System.IO;
using Classification.InstanceList;
using Math;

namespace Classification.Model
{
    public class QdaModel : LdaModel
    {
        private Dictionary<string, Matrix> _W;

        public QdaModel()
        {

        }

        /// <summary>
        /// Loads a quadratic discriminant analysis model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        private new void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            var size = LoadPriorDistribution(input);
            LoadWandW0(input, size);
            _W = new Dictionary<string, Matrix>();
            for (var i = 0; i < size; i++)
            {
                var c = input.ReadLine();
                var matrix = LoadMatrix(input);
                _W[c] = matrix;
            }

            input.Close();
        }

        /// <summary>
        /// Loads a quadratic discriminant analysis model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public QdaModel(string fileName)
        {
            Load(fileName);
        }

        /**
         * <summary> The calculateMetric method takes an {@link Instance} and a String as inputs. It multiplies Matrix Wi with Vector xi
         * then calculates the dot product of it with xi. Then, again it finds the dot product of wi and xi and returns the summation with w0i.</summary>
         *
         * <param name="instance">{@link Instance} input.</param>
         * <param name="ci">      String input.</param>
         * <returns>The result of Wi.multiplyWithVectorFromLeft(xi).dotProduct(xi) + wi.dotProduct(xi) + w0i.</returns>
         */
        protected override double CalculateMetric(Instance.Instance instance, string ci)
        {
            var xi = instance.ToVector();
            var Wi = _W[ci];
            var wi = w[ci];
            var w0I = w0[ci];
            return Wi.MultiplyWithVectorFromLeft(xi).DotProduct(xi) + wi.DotProduct(xi) + w0I;
        }

        /**
         * <summary> Training algorithm for the quadratic discriminant analysis classifier (Introduction to Machine Learning, Alpaydin, 2015).</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            w0 = new Dictionary<string, double>();
            w = new Dictionary<string, Vector>();
            _W = new Dictionary<string, Matrix>();
            var classLists = trainSet.DivideIntoClasses();

            priorDistribution = trainSet.ClassDistribution();
            for (var i = 0; i < classLists.Size(); i++)
            {
                var ci = ((InstanceListOfSameClass)classLists.Get(i)).GetClassLabel();
                var averageVector = new Vector(classLists.Get(i).ContinuousAttributeAverage());
                var classCovariance = classLists.Get(i).Covariance(averageVector);
                var determinant = classCovariance.Determinant();
                classCovariance.Inverse();

                var Wi = (Matrix)classCovariance.Clone();
                Wi.MultiplyWithConstant(-0.5);
                _W[ci] = Wi;
                var wi = classCovariance.MultiplyWithVectorFromLeft(averageVector);
                w[ci] = wi;
                var w0I = -0.5 * (wi.DotProduct(averageVector) + System.Math.Log(determinant)) +
                          System.Math.Log(priorDistribution.GetProbability(ci));
                w0[ci] = w0I;
            }

        }

        /// <summary>
        /// Loads the Qda model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the Qda model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }
    }
}