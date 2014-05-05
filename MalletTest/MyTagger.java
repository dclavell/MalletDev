import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.fst.CRFOptimizableByLabelLikelihood;
import cc.mallet.fst.CRFTrainerByValueGradients;
import cc.mallet.fst.CRFWriter;
import cc.mallet.fst.MultiSegmentationEvaluator;
import cc.mallet.fst.TransducerEvaluator;
import cc.mallet.fst.TransducerTrainer;
import cc.mallet.optimize.Optimizable;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.types.Alphabet;
import cc.mallet.types.AugmentableFeatureVector;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelSequence;

public class MyTagger {

	public static String[] features = { "HIP_CENTER", "SPINE",
			"SHOULDER_CENTER", "HEAD", "SHOULDER_LEFT", "ELBOW_LEFT",
			"WRIST_LEFT", "HAND_LEFT", "SHOULDER_RIGHT", "ELBOW_RIGHT",
			"WRIST_RIGHT", "HAND_RIGHT", "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT",
			"FOOT_LEFT", "HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT", "FOOT_RIGHT" };

	public static int K = 10;
	public static int NUM_FILES_TO_USE = 10; // This needs to be at least K
	public static boolean USING_BINARY_ANGLE_FEATURES = true;
	public static int EPSILON = 15;
	public static int ANGLE_DIFFERENCE = 5;

	public static void main(String[] args) throws Exception {
		// Generate the InstanceLists of training/testing data.
		// Pipe instancePipe = new

		File dir = new File(
				"E:\\Downloads\\MicrosoftGestureDataset\\MicrosoftGestureDataset-RC\\data");
		File[] files = dir.listFiles();
		List<File> dataFiles = new ArrayList<File>();
		for (File file : files) {
			if (file.isFile() && file.getName().contains(".csv")) {
				dataFiles.add(file);

				// Modify the data to have the right information.
				// StringBuilder sb = new StringBuilder();
				// Scanner s;
				//
				// s = new Scanner(new File(file.getAbsolutePath().replaceAll(
				// "\\.csv", ".tagstream")));
				// s.nextLine();
				// String tagInfo = " "
				// + s.nextLine().split(";")[1].split("\\s")[0];
				// System.out.println(tagInfo);
				// s.close();
				//
				// s = new Scanner(file);
				// while (s.hasNextLine()) {
				// String st = s.nextLine();
				// sb.append(st);
				// sb.append(tagInfo);
				// sb.append("\n");
				// }
				// s.close();
				//
				// FileWriter fw = new FileWriter(file);
				// fw.write(sb.toString());
				// fw.flush();
				// fw.close();

			}
		}

		// K-fold Cross-Validation:

		// -- randomize the data
		Collections.shuffle(dataFiles);

		// -- make it arbitrarily shorter so I don't run out of memory

		while (dataFiles.size() > NUM_FILES_TO_USE) {
			dataFiles.remove(0);
		}

		// -- divide into folds, train on each K-1/K, then test on the removed
		// 1/K
		for (int fold = 0; fold < K; fold++) {
			List<File> testFiles = new ArrayList<File>();
			List<File> trainFiles = new ArrayList<File>();
			for (int fileIndex = 0; fileIndex < dataFiles.size(); fileIndex++) {
				if (fileIndex % K == fold) {
					testFiles.add(dataFiles.get(fileIndex));
				} else {
					trainFiles.add(dataFiles.get(fileIndex));
				}
			}
			Pipe pTrain = new MyTaggerSentence2FeatureVectorSequence();
			Pipe pTest = new MyTaggerSentence2FeatureVectorSequence();
			pTrain.getTargetAlphabet().lookupIndex("000");
			pTrain.setTargetProcessing(true);
			pTest.getTargetAlphabet().lookupIndex("000");
			pTest.setTargetProcessing(true);
			InstanceList trainingData = new InstanceList(pTrain);
			InstanceList testData = new InstanceList(pTrain);

			// Go through all the files and add some percentage of them.
			// Do this... however many times?
			// k-means? 1/10 left for test?

			for (File file : trainFiles) {
				Reader trainingFile = new FileReader(file);
				trainingData.addThruPipe(new LineGroupIterator(trainingFile,
						Pattern.compile("^\\s*$"), true));
			}
			for (File file : testFiles) {
				Reader testFile = new FileReader(file);
				testData.addThruPipe(new LineGroupIterator(testFile, Pattern
						.compile("^\\s*$"), true));
			}
			System.out.println("Number of features in training data: "
					+ pTrain.getDataAlphabet().size());
			System.out.println(pTrain.getTargetAlphabet());
			// What are start/continue tags?
			// Needs more setup to train?
			new MyTagger().run(trainingData, testData);
		}
	}

	public void run(InstanceList trainingData, InstanceList testingData) {
		// setup:
		// CRF (model) and the state machine
		// CRFOptimizableBy* objects (terms in the objective function)
		// CRF trainer
		// evaluator and writer

		// model
		CRF crf = new CRF(trainingData.getDataAlphabet(),
				trainingData.getTargetAlphabet());
		// construct the finite state machine
		crf.addFullyConnectedStatesForLabels();
		// initialize model's weights
		crf.setWeightsDimensionAsIn(trainingData, false);

		// CRFOptimizableBy* objects (terms in the objective function)
		// objective 1: label likelihood objective
		CRFOptimizableByLabelLikelihood optLabel = new CRFOptimizableByLabelLikelihood(
				crf, trainingData);

		// CRF trainer
		Optimizable.ByGradientValue[] opts = new Optimizable.ByGradientValue[] { optLabel };
		// by default, use L-BFGS as the optimizer
		CRFTrainerByValueGradients crfTrainer = new CRFTrainerByValueGradients(
				crf, opts);

		// *Note*: labels can also be obtained from the target alphabet
		String[] labels = new String[] { "G1", "G2", "G3", "G4", "G5", "G6",
				"G7", "G8", "G9", "G10", "G11", "G12" };
		TransducerEvaluator evaluator = new MultiSegmentationEvaluator(
				new InstanceList[] { trainingData, testingData }, new String[] {
						"train", "test" }, labels, labels) {
			@Override
			public boolean precondition(TransducerTrainer tt) {
				// evaluate model every training iteration
				return true;
			}
		};
		crfTrainer.addEvaluator(evaluator);

		CRFWriter crfWriter = new CRFWriter("ner_crf.model") {
			@Override
			public boolean precondition(TransducerTrainer tt) {
				// save the trained model after training finishes
				return tt.getIteration() % Integer.MAX_VALUE == 0;
			}
		};
		crfTrainer.addEvaluator(crfWriter);

		// all setup done, train until convergence
		crfTrainer.setMaxResets(0);
		crfTrainer.train(trainingData, Integer.MAX_VALUE);
		// evaluate
		evaluator.evaluate(crfTrainer);

		// save the trained model (if CRFWriter is not used)
		// FileOutputStream fos = new FileOutputStream("ner_crf.model");
		// ObjectOutputStream oos = new ObjectOutputStream(fos);
		// oos.writeObject(crf);
	}

	public static class MyTaggerSentence2FeatureVectorSequence extends Pipe {
		// gdruck
		// Previously, there was no serialVersionUID. This is ID that would
		// have been automatically generated by the compiler. Therefore,
		// other changes should not break serialization.
		private static final long serialVersionUID = -2059308802200728625L;

		/**
		 * Creates a new <code>SimpleTaggerSentence2FeatureVectorSequence</code>
		 * instance.
		 */
		public MyTaggerSentence2FeatureVectorSequence() {
			super(new Alphabet(), new LabelAlphabet());
		}

		/**
		 * Parses a string representing a sequence of rows of tokens into an
		 * array of arrays of tokens.
		 * 
		 * @param sentence
		 *            a <code>String</code>
		 * @return the corresponding array of arrays of tokens.
		 */
		private String[][] parseSentence(String sentence) {
			String[] lines = sentence.split("\n");
			String[][] tokens = new String[lines.length][];
			for (int i = 0; i < lines.length; i++)
				tokens[i] = lines[i].split(" ");
			// Instead, parse them into the XYZ positions of each joint.

			// Joint Places:
			//
			// HIP_CENTER = 0;
			// SPINE = 1;
			// SHOULDER_CENTER = 2;
			// HEAD = 3;
			// SHOULDER_LEFT = 4;
			// ELBOW_LEFT = 5;
			// WRIST_LEFT = 6;
			// HAND_LEFT = 7;
			// SHOULDER_RIGHT = 8;
			// ELBOW_RIGHT = 9;
			// WRIST_RIGHT = 10;
			// HAND_RIGHT = 11;
			// HIP_LEFT = 12;
			// KNEE_LEFT = 13;
			// ANKLE_LEFT = 14;
			// FOOT_LEFT = 15;
			// HIP_RIGHT = 16;
			// KNEE_RIGHT = 17;
			// ANKLE_RIGHT = 18;
			// FOOT_RIGHT = 19
			//
			// Joint Connections:
			//
			// HIP_CENTER, SPINE; ...
			// SPINE, SHOULDER_CENTER; ...
			// SHOULDER_CENTER, HEAD; ...
			// % Left arm ...
			// SHOULDER_CENTER, SHOULDER_LEFT; ...
			// SHOULDER_LEFT, ELBOW_LEFT; ...
			// ELBOW_LEFT, WRIST_LEFT; ...
			// WRIST_LEFT, HAND_LEFT; ...
			// % Right arm ...
			// SHOULDER_CENTER, SHOULDER_RIGHT; ...
			// SHOULDER_RIGHT, ELBOW_RIGHT; ...
			// ELBOW_RIGHT, WRIST_RIGHT; ...
			// WRIST_RIGHT, HAND_RIGHT; ...
			// % Left leg ...
			// HIP_CENTER, HIP_LEFT; ...
			// HIP_LEFT, KNEE_LEFT; ...
			// KNEE_LEFT, ANKLE_LEFT; ...
			// ANKLE_LEFT, FOOT_LEFT; ...
			// % Right leg ...
			// HIP_CENTER, HIP_RIGHT; ...
			// HIP_RIGHT, KNEE_RIGHT; ...
			// KNEE_RIGHT, ANKLE_RIGHT; ...
			// ANKLE_RIGHT, FOOT_RIGHT ...
			return tokens;
		}

		public Instance pipe(Instance carrier) {
			Object inputData = carrier.getData();
			// TODO: features are actually the angles and positions, not
			// numbers!
			// TODO: Figure out how to make your own Alphabet

			Set<String> naNFeatures = new HashSet<String>();
			Alphabet featureAlphabet = getDataAlphabet();
			LabelAlphabet labels;
			LabelSequence target = null;
			String[][] tokens;
			double max = 0.0;
			if (inputData instanceof String)
				tokens = parseSentence((String) inputData);
			else if (inputData instanceof String[][])
				tokens = (String[][]) inputData;
			else
				throw new IllegalArgumentException(
						"Not a String or String[][]; got " + inputData);
			FeatureVector[] fvs = new FeatureVector[tokens.length];
			if (isTargetProcessing()) {
				labels = (LabelAlphabet) getTargetAlphabet();
				target = new LabelSequence(labels, tokens.length);
			}
			for (int l = 0; l < tokens.length; l++) {
				int nFeatures;
				if (isTargetProcessing()) {
					if (tokens[l].length < 1)
						throw new IllegalStateException(
								"Missing label at line " + l + " instance "
										+ carrier.getName());
					nFeatures = tokens[l].length - 1;
					target.add(tokens[l][nFeatures]);
				} else
					nFeatures = tokens[l].length;
				ArrayList<Integer> featureIndices = new ArrayList<Integer>();

				// I'll ignore this
				String timestamp = tokens[l][0];

				double[] positionValues = new double[60];
				ArrayList<String> featureNames = new ArrayList<String>();

				// Generate a list of the feature names:

				for (int f = 0; f < 20; f++) {
					featureNames.add(features[f] + "_X");
					featureNames.add(features[f] + "_Y");
					featureNames.add(features[f] + "_Z");
				}

				for (int f = 1; f < 20; f++) {
					for (int g = f + 1; g < 20; g++) {
						featureNames.add(features[f] + "_" + features[g]
								+ "_angle");
					}
				}

				// These are the x y and z coordinates.
				for (int f = 0; f < 20; f++) {
					/* Add the position feature indices */
					int xIndex = featureAlphabet
							.lookupIndex(features[f] + "_X");
					featureIndices.add(xIndex);
					positionValues[f * 3] = Double
							.parseDouble(tokens[l][f * 3 + 1]);
					int yIndex = featureAlphabet
							.lookupIndex(features[f] + "_Y");
					featureIndices.add(yIndex);
					positionValues[f * 3 + 1] = Double
							.parseDouble(tokens[l][f * 3 + 2]);
					int zIndex = featureAlphabet
							.lookupIndex(features[f] + "_Z");
					featureIndices.add(zIndex);
					positionValues[f * 3 + 2] = Double
							.parseDouble(tokens[l][f * 3 + 3]);
				}

				double[] angleFeatureValues = new double[400 * (360 / ANGLE_DIFFERENCE + 1)];
				int angleValueIndex = 0;
				for (int f = 1; f < 20; f++) {
					for (int g = f + 1; g < 20; g++) {
						// Generate all angle names.
						int angleIndex = featureAlphabet
								.lookupIndex(features[f] + "_" + features[g]
										+ "_angle");
						featureIndices.add(angleIndex);

						// Calculate angle
						// cos a = v . w / ( |v| * |w| )
						int h = 0; // The index of HIP_CENTER
						double fUnitX, fUnitY, fUnitZ, gUnitX, gUnitY, gUnitZ, hUnitX, hUnitY, hUnitZ;
						fUnitX = positionValues[f * 3];
						fUnitY = positionValues[f * 3 + 1];
						fUnitZ = positionValues[f * 3 + 2];
						gUnitX = positionValues[g * 3];
						gUnitY = positionValues[g * 3 + 1];
						gUnitZ = positionValues[g * 3 + 2];
						hUnitX = positionValues[h * 3];
						hUnitY = positionValues[h * 3 + 1];
						hUnitZ = positionValues[h * 3 + 2];

						double dotProduct = (fUnitX - hUnitX)
								* (gUnitX - hUnitX) + (fUnitY - hUnitY)
								* (gUnitY - hUnitY) + (fUnitZ - hUnitZ)
								* (gUnitZ - hUnitZ);

						double fCard = Math.sqrt((fUnitX - hUnitX)
								* (fUnitX - hUnitX) + (fUnitY - hUnitY)
								* (fUnitY - hUnitY) + (fUnitZ - hUnitZ)
								* (fUnitZ - hUnitZ));
						double gCard = Math.sqrt((gUnitX - hUnitX)
								* (gUnitX - hUnitX) + (gUnitY - hUnitY)
								* (gUnitY - hUnitY) + (gUnitZ - hUnitZ)
								* (gUnitZ - hUnitZ));

						double cardMult = fCard * gCard;
						// Uncomment to try and prevent 0 muddling
						// cardMult = (cardMult == 0.0) ? Double.MIN_NORMAL
						// : cardMult;

						double angle = Math.acos(dotProduct / cardMult);

						if (new Double(angle).equals(Double.NaN)) {
							angle = 1.0;
						}

						if (new Double(angle).equals(Double.NaN)
								|| Double.isInfinite(angle)) {
							naNFeatures.add(features[f] + "_" + features[g]
									+ "_angle");
						}

						if (Double.isInfinite(angle)) {
							angle = Double.MAX_VALUE;
							// System.out.println(features[f] + "_" +
							// features[g]
							// + "_angle");
						}

						angleFeatureValues[angleValueIndex++] = angle;

						// If we're using the new binary features
						if (USING_BINARY_ANGLE_FEATURES) {
							for (int i = 0; i < 360; i += ANGLE_DIFFERENCE) {
								if (i + EPSILON >= 360) {
									if (angle >= i
											|| angle <= (i + EPSILON) % 360) {
										int binaryFeatureIndex = featureAlphabet
												.lookupIndex(features[f] + "_"
														+ features[g]
														+ "_angle_" + i + "-"
														+ (i + 15));
										featureIndices.add(binaryFeatureIndex);
										angleFeatureValues[angleValueIndex++] = 1;
									}
								} else if (angle >= i && angle <= i + EPSILON) {
									int binaryFeatureIndex = featureAlphabet
											.lookupIndex(features[f] + "_"
													+ features[g] + "_angle_"
													+ i + "-" + (i + 15));
									featureIndices.add(binaryFeatureIndex);
									angleFeatureValues[angleValueIndex++] = 1;
								}
							}
						}
					}
				}
				// System.out.println();

				// for (int f = 0; f < nFeatures; f++) {
				//
				// int featureIndex = features.lookupIndex(tokens[l][f]);
				// // gdruck
				// // If the data alphabet's growth is stopped, featureIndex
				// // will be -1. Ignore these features.
				// if (featureIndex >= 0) {
				// featureIndices.add(featureIndex);
				// }
				// }
				int[] featureIndicesArr = new int[featureIndices.size()];
				for (int index = 0; index < featureIndices.size(); index++) {
					featureIndicesArr[index] = featureIndices.get(index);
				}

				// Use actual values! And change them or something!
				// Take advantage of the Augmentable Feature Vector!

				// Aggregate the different feature values.
				double[] values = new double[featureIndicesArr.length];

				for (int i = 0; i < positionValues.length; i++) {
					values[i] = positionValues[i];
				}

				for (int i = positionValues.length; i < featureIndicesArr.length; i++) {
					values[i] = angleFeatureValues[i];
				}

				// System.out.println("Feature Alphabet:\n\n" +
				// featureAlphabet.toString());
				// System.out.println("Values:\n\n" + Arrays.toString(values));
				fvs[l] = new AugmentableFeatureVector(featureAlphabet,
						featureIndicesArr, values, featureIndicesArr.length);
			}
			// System.out.println("Max value: " + max);
			// System.out.println(naNFeatures);
			carrier.setData(new FeatureVectorSequence(fvs));
			if (isTargetProcessing())
				carrier.setTarget(target);
			else
				carrier.setTarget(new LabelSequence(getTargetAlphabet()));

			// System.out.println();
			return carrier;
		}
	}

}
