using Classification.Classifier;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class LinearPerceptronTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var linearPerceptron = new LinearPerceptron();
            var linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
            linearPerceptron.Train(iris.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(8.67, 100 * linearPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            linearPerceptronParameter = new LinearPerceptronParameter(1, 0.001, 0.99, 0.2, 100);
            linearPerceptron.Train(bupa.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(31.88, 100 * linearPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
            linearPerceptron.Train(dermatology.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(3.28, 100 * linearPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}