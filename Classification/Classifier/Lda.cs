using System.Collections.Generic;
using Classification.InstanceList;
using Classification.Model;
using Math;

namespace Classification.Classifier
{
    public class Lda : Classifier
    {
        /**
         * <summary> Training algorithm for the linear discriminant analysis classifier (Introduction to Machine Learning, Alpaydin, 2015).</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            Vector averageVector;
            var w0 = new Dictionary<string, double>();
            var w = new Dictionary<string, Vector>();
            var priorDistribution = trainSet.ClassDistribution();
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
                var w0i = -0.5 * wi.DotProduct(averageVector) + System.Math.Log(priorDistribution.GetProbability(ci));
                w0[ci] = w0i;
            }

            model = new LdaModel(priorDistribution, w, w0);
        }
    }
}