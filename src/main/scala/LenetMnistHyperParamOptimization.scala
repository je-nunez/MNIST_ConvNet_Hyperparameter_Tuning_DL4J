package mainapp

// A Scala version of the Convolutional Neural Network for MNIST written in Java using the
// DeepLearning4J library, modified further to use DL4J's Arbiter classes which implement
// hyperparameter tuning.
//
//    https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java
//
// The following users are credited as the collaborators on above file (in comments in file above):
//   LenetMnistExample.java:   Created by agibsonccc on 9/16/15.
//   LenetMnistExample.java:   Modified by dmichelin on 12/10/2016 to add documentation

import scala.collection.JavaConverters._

import java.util.HashMap
import java.util.Map
import java.util.concurrent.TimeUnit

import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.iterator.MultipleEpochsIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
// import org.deeplearning4j.nn.conf.LearningRatePolicy
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.weights.WeightInit

import org.deeplearning4j.arbiter.DL4JConfiguration
import org.deeplearning4j.arbiter.MultiLayerSpace
import org.deeplearning4j.arbiter.layers.ConvolutionLayerSpace
import org.deeplearning4j.arbiter.layers.DenseLayerSpace
import org.deeplearning4j.arbiter.layers.OutputLayerSpace
import org.deeplearning4j.arbiter.layers.SubsamplingLayerSpace
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner
import org.deeplearning4j.arbiter.scoring.impl.TestSetAccuracyScoreFunction
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator

import org.omg.PortableInterceptor.SYSTEM_EXCEPTION


object LenetMnistHyperParamOptimization {

  val batchSize = 64         // Test batch size
  val seed = 123

  def main(args: Array[String]): Unit = {
    val nChannels = 1          // Number of input channels
    val outputNum = 10         // number of possible outcomes (number of MNIST digits)
    val iterations = 1         // Number of training iterations

    ReflectionsHelper.registerUrlTypes()

    Nd4j.ENFORCE_NUMERICAL_STABILITY = true

    // some hyperparameters of the MNIST convergent neural network to tune with Monte Carlo methods.
    // We call it the universe of values where the Monte Carlo method will randomly sample its values.
    val l2Universe = new ContinuousParameterSpace(0.000005, 0.0001)
    val globalLearningRateUniverse = new ContinuousParameterSpace(0.001, 0.1)
    val globalBiasLearningRateUniverse = new ContinuousParameterSpace(0.001, 0.01)
    val updatersUniverse = new DiscreteParameterSpace(Updater.NESTEROVS)
    val convolLayer1nOutUniverse = new IntegerParameterSpace(15, 25)
    val convolLayer3nOutUniverse = new IntegerParameterSpace(40, 60)
    val denseLayer5nOutUniverse = new IntegerParameterSpace(400, 600)

    println("Building templated CNN model for hyperparameter tuning....")
    val parameterizedNN =
      new MultiLayerSpace.Builder()
        .seed(seed)
        .iterations(iterations)
        .regularization(true)
        .l2(l2Universe)
        .learningRate(globalLearningRateUniverse)
        .biasLearningRate(globalBiasLearningRateUniverse)
        // .learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
        .weightInit(WeightInit.XAVIER)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(updatersUniverse)
        .addLayer(new ConvolutionLayerSpace.Builder()
                    .kernelSize(5, 5)
                    // nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                    .nIn(nChannels)
                    .stride(1, 1)
                    .nOut(convolLayer1nOutUniverse)
                    .activation(Activation.IDENTITY)
                    .build()
              )
        .addLayer(new SubsamplingLayerSpace.Builder()
                    .poolingType(SubsamplingLayer.PoolingType.MAX)
                    // .poolingType(SubsamplingLayer.PoolingType.PNORM)
                    // .pNorm(2)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
              )
        .addLayer(new ConvolutionLayerSpace.Builder()
                    .kernelSize(5, 5)
                    // Note that nIn need not be specified in later layers
                    .stride(1, 1)
                    .nOut(convolLayer3nOutUniverse)
                    .activation(Activation.IDENTITY)
                    .build()
              )
        .addLayer(new SubsamplingLayerSpace.Builder()
                    .poolingType(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
              )
        .addLayer(new DenseLayerSpace.Builder()
                    .activation(Activation.RELU)
                    .nOut(denseLayer5nOutUniverse)
                    .build()
              )
        .addLayer(new OutputLayerSpace.Builder()
                    .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build()
              )
        .setInputType(InputType.convolutionalFlat(28, 28, 1))
        .backprop(true)
        .pretrain(false)
        .build()


    val commands = new HashMap[String, Object]()

    val nnCandidateGenerator = new RandomSearchGenerator(parameterizedNN, commands)

    // stop the search either after checking 10'000 candidate models, or after such run-time
    val stopAfterMinutes = 120
    val searchStopConditions = List(new MaxCandidatesCondition(10000),
                                    new MaxTimeCondition(stopAfterMinutes, TimeUnit.MINUTES)
                               )

    val searchScoreFunction = new TestSetAccuracyScoreFunction()

    val dataProvider = MnistDataSetProvider

    val optimizationConfig =
      new OptimizationConfiguration.Builder()
        .terminationConditions(searchStopConditions.asJava)
        .candidateGenerator(nnCandidateGenerator)
        .scoreFunction(searchScoreFunction)
        .dataProvider(dataProvider)
        .build()

    // Set up search to be executed locally on this (local) computer
    val optimRunner =
      new LocalOptimizationRunner(optimizationConfig, new MultiLayerNetworkTaskCreator())

    optimRunner.execute()

    val indexBestModel = optimRunner.bestScoreCandidateIndex
    val allModels = optimRunner.getResults

    if (indexBestModel != -1) {
      println(s"""Best model index: $indexBestModel,
                  |with score: ${optimRunner.bestScore}
                  |found among ${optimRunner.numCandidatesCompleted} candidate models completed"""
              .stripMargin.replaceAll("\n", " ")
             )
    } else {
      println(s"Best model couldn't be found. It seems that $stopAfterMinutes minutes weren't enough.")
    }

  }


  // taken from "arbiter-deeplearning4j"'s "MNISTOptimizationTest.java" file

  object MnistDataSetProvider extends DataProvider {

    override def trainData(dataParameters: Map[String, Object]): DataSetIterator = {
      try {
        if (dataParameters == null || dataParameters.isEmpty) {
          new MnistDataSetIterator(batchSize, 10000, false, true, true, seed)
        }
        if (dataParameters.containsKey("batchsize")) {
          val b = dataParameters.get("batchsize").asInstanceOf[Int]
          new MnistDataSetIterator(b, 10000, false, true, true, seed)
        }
        new MnistDataSetIterator(batchSize, 10000, false, true, true, seed)
      } catch {
        case e: Exception => throw new RuntimeException(e)
      }
    }

    override def testData(dataParameters: Map[String, Object]): DataSetIterator = {
      trainData(dataParameters)
    }

    override def getDataType: Class[_] = {
      classOf[DataSetIterator]
    }

    override def toString: String = {
      "MnistDataSetProvider()"
    }
  }

}
