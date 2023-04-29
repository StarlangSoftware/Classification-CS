using Classification.Model;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class KMeans : Classifier
    {
        /**
         * <summary> Training algorithm for K-Means classifier. K-Means finds the mean of each class for training.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">distanceMetric: distance metric used to calculate the distance between two instances.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            var priorDistribution = trainSet.ClassDistribution();
            var classMeans = new InstanceList.InstanceList();
            var classLists = trainSet.DivideIntoClasses();
            for (var i = 0; i < classLists.Size(); i++)
            {
                classMeans.Add(classLists.Get(i).Average());
            }

            model = new KMeansModel(priorDistribution, classMeans, ((KMeansParameter) parameters).GetDistanceMetric());
        }

        public override void LoadModel(string fileName)
        {
            model = new KMeansModel(fileName);
        }
    }
}