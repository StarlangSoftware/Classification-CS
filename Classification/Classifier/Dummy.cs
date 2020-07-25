using Classification.Model;

namespace Classification.Classifier
{
    public class Dummy : Classifier
    {
        /**
         * <summary> Training algorithm for the dummy classifier. Actually dummy classifier returns the maximum occurring class in
         * the training data, there is no training.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            model = new DummyModel(trainSet);
        }
    }
}