using System.Collections.Generic;
using Classification.InstanceList;
using Classification.Model;
using Math;

namespace Classification.Classifier
{
    public class Qda : Classifier
    {
        /**
         * <summary> Training algorithm for the quadratic discriminant analysis classifier (Introduction to Machine Learning, Alpaydin, 2015).</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            var w0 = new Dictionary<string, double>();
            var w = new Dictionary<string, Vector>();
            var W = new Dictionary<string, Matrix>();
            var classLists = trainSet.DivideIntoClasses();

            var priorDistribution = trainSet.ClassDistribution();
            for (var i = 0; i < classLists.Size(); i++)
            {
                var ci = ((InstanceListOfSameClass) classLists.Get(i)).GetClassLabel();
                var averageVector = new Vector(classLists.Get(i).ContinuousAttributeAverage());
                var classCovariance = classLists.Get(i).Covariance(averageVector);
                var determinant = classCovariance.Determinant();
                classCovariance.Inverse();

                var Wi = (Matrix) classCovariance.Clone();
                Wi.MultiplyWithConstant(-0.5);
                W[ci] = Wi;
                var wi = classCovariance.MultiplyWithVectorFromLeft(averageVector);
                w[ci] = wi;
                var w0i = -0.5 * (wi.DotProduct(averageVector) + System.Math.Log(determinant)) +
                          System.Math.Log(priorDistribution.GetProbability(ci));
                w0[ci] = w0i;
            }

            model = new QdaModel(priorDistribution, W, w, w0);
        }

        public override void LoadModel(string fileName)
        {
            model = new QdaModel(fileName);
        }
    }
}