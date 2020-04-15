using Classification.Parameter;
using Math;

namespace Classification.Model
{
    public class AutoEncoderModel : NeuralNetworkModel
    {
        private Matrix _V, _W;

        /**
         * <summary> The allocateWeights method takes an integer number and sets layer weights of W and V matrices according to given number.</summary>
         *
         * <param name="H">Integer input.</param>
         */
        private void AllocateWeights(int H)
        {
            _W = AllocateLayerWeights(H, d + 1);
            _V = AllocateLayerWeights(K, H + 1);
        }

        /**
         * <summary> The {@link AutoEncoderModel} method takes two {@link InstanceList}s as inputs; train set and validation set. First it allocates
         * the weights of W and V matrices using given {@link MultiLayerPerceptronParameter} and takes the clones of these matrices as the bestW and bestV.
         * Then, it gets the epoch and starts to iterate over them. First it shuffles the train set and tries to find the new W and V matrices.
         * At the end it tests the autoencoder with given validation set and if its performance is better than the previous one,
         * it reassigns the bestW and bestV matrices. Continue to iterate with a lower learning rate till the end of an episode.</summary>
         *
         * <param name="trainSet">     {@link InstanceList} to use as train set.</param>
         * <param name="validationSet">{@link InstanceList} to use as validation set.</param>
         * <param name="parameters">   {@link MultiLayerPerceptronParameter} is used to get the parameters.</param>
         */
        public AutoEncoderModel(InstanceList.InstanceList trainSet, InstanceList.InstanceList validationSet,
            MultiLayerPerceptronParameter parameters) : base(trainSet)
        {
            K = trainSet.Get(0).ContinuousAttributeSize();
            AllocateWeights(parameters.GetHiddenNodes());
            Matrix bestW = (Matrix) _W.Clone();
            Matrix bestV = (Matrix) _V.Clone();
            Performance.Performance bestPerformance = new Performance.Performance(double.MaxValue);
            int epoch = parameters.GetEpoch();
            double learningRate = parameters.GetLearningRate();
            for (var i = 0; i < epoch; i++)
            {
                trainSet.Shuffle(parameters.GetSeed());
                for (var j = 0; j < trainSet.Size(); j++)
                {
                    CreateInputVector(trainSet.Get(j));
                    r = trainSet.Get(j).ToVector();
                    Vector hidden = CalculateHidden(x, _W);
                    Vector hiddenBiased = hidden.Biased();
                    y = _V.MultiplyWithVectorFromRight(hiddenBiased);
                    Vector rMinusY = r.Difference(y);
                    Matrix deltaV = rMinusY.Multiply(hiddenBiased);
                    Vector oneMinusHidden = CalculateOneMinusHidden(hidden);
                    Vector tmph = _V.MultiplyWithVectorFromLeft(rMinusY);
                    tmph.Remove(0);
                    Vector tmpHidden = oneMinusHidden.ElementProduct(hidden.ElementProduct(tmph));
                    Matrix deltaW = tmpHidden.Multiply(x);
                    deltaV.MultiplyWithConstant(learningRate);
                    _V.Add(deltaV);
                    deltaW.MultiplyWithConstant(learningRate);
                    _W.Add(deltaW);
                }

                Performance.Performance currentPerformance = TestAutoEncoder(validationSet);
                if (currentPerformance.GetErrorRate() < bestPerformance.GetErrorRate())
                {
                    bestPerformance = currentPerformance;
                    bestW = (Matrix) _W.Clone();
                    bestV = (Matrix) _V.Clone();
                }

                learningRate *= 0.95;
            }

            _W = bestW;
            _V = bestV;
        }

        /**
         * <summary> The testAutoEncoder method takes an {@link InstanceList} as an input and tries to predict a value and finds the difference with the
         * actual value for each item of that InstanceList. At the end, it returns an error rate by finding the mean of total errors.</summary>
         *
         * <param name="data">{@link InstanceList} to use as validation set.</param>
         * <returns>Error rate by finding the mean of total errors.</returns>
         */
        public Performance.Performance TestAutoEncoder(InstanceList.InstanceList data)
        {
            double total = data.Size();
            var error = 0.0;
            for (var i = 0; i < total; i++)
            {
                y = PredictInput(data.Get(i));
                r = data.Get(i).ToVector();
                error += r.Difference(y).DotProduct();
            }

            return new Performance.Performance(error / total);
        }

        /**
         * <summary> The predictInput method takes an {@link Instance} as an input and calculates a forward single hidden layer and returns the predicted value.</summary>
         *
         * <param name="instance">{@link Instance} to predict.</param>
         * <returns>Predicted value.</returns>
         */
        private Vector PredictInput(Instance.Instance instance)
        {
            CreateInputVector(instance);
            CalculateForwardSingleHiddenLayer(_W, _V);
            return y;
        }

        /**
         * <summary> The calculateOutput method calculates a forward single hidden layer.</summary>
         */
        protected override void CalculateOutput()
        {
            CalculateForwardSingleHiddenLayer(_W, _V);
        }
    }
}