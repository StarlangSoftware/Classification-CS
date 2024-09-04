using System.Collections.Generic;
using System.IO;
using Classification.InstanceList;
using Math;

namespace Classification.Model
{
    public class LdaModel : GaussianModel
    {
        protected Dictionary<string, double> w0;
        protected Dictionary<string, Vector> w;
        
        public LdaModel()
        {
            
        }
        
        /// <summary>
        /// Loads w0 and w hash maps from an input file. The number of items in the hash map is given by the parameter size.
        /// </summary>
        /// <param name="input">Input file</param>
        /// <param name="size">Number of items in the hash map read.</param>
        protected void LoadWandW0(StreamReader input, int size)
        {
            w0 = new Dictionary<string, double>();
            for (var i = 0; i < size; i++)
            {
                var line = input.ReadLine();
                var items = line.Split(" ");
                w0[items[0]] =  double.Parse(items[1]);
            }

            w = LoadVectors(input, size);
        }

        /// <summary>
        /// Loads a Linear Discriminant Analysis model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            var size = LoadPriorDistribution(input);
            LoadWandW0(input, size);
            input.Close();
        }

        /// <summary>
        /// Loads a Linear Discriminant Analysis model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public LdaModel(string fileName)
        {
            Load(fileName);
        }

        /**
         * <summary> The calculateMetric method takes an {@link Instance} and a String as inputs. It returns the dot product of given Instance
         * and wi plus w0i.</summary>
         *
         * <param name="instance">{@link Instance} input.</param>
         * <param name="ci">      String input.</param>
         * <returns>The dot product of given Instance and wi plus w0i.</returns>
         */
        protected override double CalculateMetric(Instance.Instance instance, string ci)
        {
            var xi = instance.ToVector();
            var wi = w[ci];
            var w0I = w0[ci];
            return wi.DotProduct(xi) + w0I;
        }
        
        /**
         * <summary> Training algorithm for the linear discriminant analysis classifier (Introduction to Machine Learning, Alpaydin, 2015).</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            Vector averageVector;
            w0 = new Dictionary<string, double>();
            w = new Dictionary<string, Vector>();
            priorDistribution = trainSet.ClassDistribution();
            var classLists = trainSet.DivideIntoClasses();
            var covariance = new Matrix(trainSet.Get(0).ContinuousAttributeSize(),
                trainSet.Get(0).ContinuousAttributeSize());
            for (var i = 0; i < classLists.Size(); i++)
            {
                averageVector = new Vector(classLists.Get(i).ContinuousAttributeAverage());
                var classCovariance = classLists.Get(i).Covariance(averageVector);
                classCovariance.MultiplyWithConstant(classLists.Get(i).Size() - 1);
                covariance.Add(classCovariance);
            }

            covariance.DivideByConstant(trainSet.Size() - classLists.Size());
            covariance.Inverse();

            for (var i = 0; i < classLists.Size(); i++)
            {
                var ci = ((InstanceListOfSameClass) classLists.Get(i)).GetClassLabel();
                averageVector = new Vector(classLists.Get(i).ContinuousAttributeAverage());
                var wi = covariance.MultiplyWithVectorFromRight(averageVector);
                w[ci] = wi;
                var w0I = -0.5 * wi.DotProduct(averageVector) + System.Math.Log(priorDistribution.GetProbability(ci));
                w0[ci] = w0I;
            }

        }

        /// <summary>
        /// Loads the Lda model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the Lda model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }

    }
}