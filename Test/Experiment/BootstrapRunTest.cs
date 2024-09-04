using Classification.DistanceMetric;
using Classification.Experiment;
using Classification.Model;
using Classification.Model.DecisionTree;
using Classification.Parameter;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Experiment
{
    public class BootstrapRunTest : ClassifierTest
    {
        [Test]
        public void TestExecute()
        {
            var bootstrapRun = new BootstrapRun(50);
            var experimentPerformance =
                bootstrapRun.Execute(new Classification.Experiment.Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), iris));
            Assert.AreEqual(3.68, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                bootstrapRun.Execute(new Classification.Experiment.Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), tictactoe));
            Assert.AreEqual(13.33, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                bootstrapRun.Execute(new Classification.Experiment.Experiment(new KnnModel(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
            Assert.AreEqual(24.26, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new KnnModel(),
                new KnnParameter(1, 3, new EuclidianDistance()), dermatology));
            Assert.AreEqual(7.92, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new LdaModel(), new Parameter(1), bupa));
            Assert.AreEqual(32.25, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new LdaModel(), new Parameter(1), dermatology));
            Assert.AreEqual(2.61, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptronModel(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
            Assert.AreEqual(4.07, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptronModel(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
            Assert.AreEqual(2.92, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new NaiveBayesModel(), new Parameter(1), car));
            Assert.AreEqual(14.19, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new NaiveBayesModel(), new Parameter(1), nursery));
            Assert.AreEqual(9.72, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                bootstrapRun.Execute(new Classification.Experiment.Experiment(new BaggingModel(), new BaggingParameter(1, 50), tictactoe));
            Assert.AreEqual(3.12, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                bootstrapRun.Execute(new Classification.Experiment.Experiment(new BaggingModel(), new BaggingParameter(1, 50), car));
            Assert.AreEqual(3.05, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new DummyModel(), new Parameter(1), nursery));
            Assert.AreEqual(66.78, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = bootstrapRun.Execute(new Classification.Experiment.Experiment(new DummyModel(), new Parameter(1), iris));
            Assert.AreEqual(66.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
        }
    }
}