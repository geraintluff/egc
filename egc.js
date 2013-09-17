(function (api) {

	var TINY_AMOUNT = Number.MIN_VALUE;

	function GaussianEstimator(varCount) {
		if (!(this instanceof GaussianEstimator)) {
			return new GaussianEstimator(varCount);
		}
		this.varCount = varCount;
		this.sums = [];
		this.products = [];
		this.total = 0;
		for (var i = 0; i < varCount; i++) {
			this.sums[i] = 0;
			this.products[i] = [];
			for (var j = 0; j <= i; j++) {
				this.products[i].push(0);
			}
		}

		/*
		for (var i = 0; i < varCount + 1; i++) {
			var vector = [];
			while (vector.length < varCount) {
				vector.push(Math.random());
			}
			this.addObservation(vector, TINY_AMOUNT);
		}
		*/
	}
	GaussianEstimator.prototype = {
		mutate: function (accuracy, initialPoints) {
			initialPoints = initialPoints || [];
			var estimator = new GaussianEstimator(this.varCount);
			if (this.total == 0) {
				while (initialPoints.length > 0) {
					estimator.addObservation(initialPoints.pop(), 1/accuracy);
				}
				return estimator;
			}
			var floorAccuracy = Math.floor(accuracy);
			for (var i = 0; i < floorAccuracy; i++) {
				var point = initialPoints.length ? initialPoints.pop() : this.random();
				estimator.addObservation(point, 1/accuracy);
			}
			if (floorAccuracy < accuracy) {
				var point = initialPoints.length ? initialPoints.pop() : this.random();
				estimator.addObservation(point, (1 - floorAccuracy/accuracy));
			}
			return estimator;
		},
		addObservation: function (vector, weight) {
			var varCount = this.varCount;
			if (vector.length !== varCount) {
				throw new Error('Vector length does not match GaussianEstimator size');
			}
			if (weight == undefined) {
				weight = 1;
			}
			
			this.total += weight;
			for (var i = 0; i < varCount; i++) {
				this.sums[i] += vector[i]*weight;
				for (var j = 0; j <= i; j++) {
					this.products[i][j] += vector[i]*vector[j]*weight;
				}
			}
		},
		mean: function (i) {
			if (this.total == 0) {
				return 50;
			}
			return this.sums[i]/this.total;
		},
		covar: function (i, j) {
			if (this.total == 0) {
				return 0;
			}
			if (i > j) {
				var tmp = i;
				i = j;
				j = tmp;
			}
			return this.products[j][i]/this.total - (this.sums[i]/this.total)*(this.sums[j]/this.total);
		},
		likelihood: function (vector) {
			var varCount = this.varCount;
			if (vector.length !== varCount) {
				throw new Error('Given length does not match GaussianEstimator size');
			}

			var given = [];
			for (var i = 0; i < varCount; i++) {
				given[i] = null;
			}
			var likelihood = 1;
			for (var i = 0; i < varCount; i++) {
				if (vector[i] != null) {
					var dist = this.collapse(i, given);
					likelihood *= dist.likelihood(vector[i]);
					given[i] = vector[i];
				}
			}
			return likelihood;
		},
		collapse: function (index, given) {
			if (this.total == 0) {
				return new Gaussian1D(0, 0);
			}
			if (!given) {
				return new Gaussian1D(this.mean(index), this.covar(index, index));
			}

			var varCount = this.varCount;
			if (given.length !== varCount) {
				throw new Error('Given length does not match GaussianEstimator size');
			}
			var mean = this.mean(index);
			var variance = this.covar(index, index);
			for (var i = 0; i < varCount; i++) {
				if (given[i] == null) {
					continue;
				}
				var iVariance = this.covar(i, i);
				if (iVariance <= 0) {
					// Floating-point errors ahoy!  Well, either that or insufficient data points.
					// If it were *actually* zero, then `covar` below would also be zero - basically, this dimension is constant
					continue;
				}
				var covar = this.covar(index, i);

				var distanceFromMean = given[i] - this.mean(i);
				mean += distanceFromMean*covar/iVariance;
				variance -= covar*covar/iVariance;
			}
			// Use Math.max() just in case a floating-point error pushes it below 0
			return new Gaussian1D(mean, Math.max(0, variance));
		},
		random: function (given) {
			var varCount = this.varCount;
			var result = [];
			for (var i = 0; i < varCount; i++) {
				result[i] = given ? given[i] : null;
			}
			for (var i = 0; i < varCount; i++) {
				if (result[i] == null) {
					var distribution = this.collapse(i, result);
					result[i] = distribution.random();
				}
			}
			return result;
		}
	};

	function Gaussian1D(mean, variance) {
		if (!(this instanceof Gaussian1D)) {
			return new Gaussian1D(mean, variance);
		}
		this.mean = mean;
		this.variance = variance;
		this.sd = Math.sqrt(variance);
	}
	var nextGaussianUnitValue = null;
	Gaussian1D.unitRandom = function () {
		if (nextGaussianUnitValue) {
			var result = nextGaussianUnitValue;
			nextGaussianUnitValue = null;
			return result;
		}
		while (true) {
			var u = Math.random()*2 - 1, v = Math.random()*2 - 1;
			var s = u*u + v*v;
			if (s == 0 || s >= 1) {
				continue;
			}
			var factor = Math.sqrt(-2*Math.log(s)/s);
			nextGaussianUnitValue = v*factor;
			return u*factor;
		}
	};
	Gaussian1D.prototype = {
		random: function () {
			return this.mean + this.sd*Gaussian1D.unitRandom();
		},
		likelihood: function (value) {
			if (this.variance <= 0) {
				return 0;
			}
			return 1/Math.sqrt(this.variance*2*Math.PI)*Math.exp(-(value - this.mean)*(value - this.mean)/(2*this.variance));
		}
	};

	function MultiEstimator(varCount, populationCount, confidenceLimit, minimumConfidence) {
		if (!(this instanceof MultiEstimator)) {
			return new MultiEstimator(varCount, populationCount);
		}
		this.varCount = varCount;
		this.populationCount;
		this.estimators = [];
		this.confidence = [];
		this.minimumConfidence = minimumConfidence || 0;
		for (var i = 0; i < populationCount; i++) {
			var estimator = new GaussianEstimator(varCount);
			// Initialise to a random point
			this.estimators.push(estimator);
			this.confidence.push(this.minimumConfidence);
		}
		this.mutationProb = 1;
		this.mutationAccuracy = 1*this.varCount + 1;
		this.confidenceLimit = confidenceLimit || populationCount;
		this.childConfidenceProportion = 0.1;
	}
	MultiEstimator.prototype = {
		killMutate: function (initialPoints) {
			var weakestIndex = 0;
			var weakestConfidence = Number.MAX_VALUE;
			var totalConfidence = 0;
			for (var i = 0; i < this.estimators.length; i++) {
				var confidence = this.confidence[i];
				totalConfidence += confidence;
				if (confidence < weakestConfidence) {
					weakestIndex = i;
					weakestConfidence = confidence;
				}
			}
			var mutationCandidateThreshhold = Math.random()*totalConfidence;
			var cumulative = 0;
			for (var i = 0; i < this.estimators.length; i++) {
				cumulative += this.confidence[i];
				if (cumulative >= mutationCandidateThreshhold) {
					var estimator = this.estimators[i];
					var mutated = estimator.mutate(this.mutationAccuracy, initialPoints);
					this.estimators[weakestIndex] = mutated;
					var stolenConfidence = this.confidence[i]*this.childConfidenceProportion;
					this.confidence[i] -= stolenConfidence;
					this.confidence[weakestIndex] = stolenConfidence;
					return;
				}
			}
		},
		likelihood: function (given) {
			var totalLikelihood = 0;
			var totalConfidence = 0;
			for (var i = 0; i < this.estimators.length; i++) {
				var estimator = this.estimators[i];
				var confidence = this.confidence[i];
				totalLikelihood += confidence*estimator.likelihood(given);
				totalConfidence += confidence;
			}
			return totalLikelihood/totalConfidence;
		},
		random: function (given) {
			var weights = [];
			var totalWeight = 0;
			for (var i = 0; i < this.estimators.length; i++) {
				var estimator = this.estimators[i];
				var confidence = this.confidence[i];
				weights[i] = estimator.likelihood(given)*confidence;
				totalWeight += weights[i];
			}
			var randomWeight = Math.random()*totalWeight;
			var cumulativeWeight = 0;
			for (var i = 0; i < this.estimators.length; i++) {
				var estimator = this.estimators[i];
				cumulativeWeight += weights[i];
				if (cumulativeWeight >= randomWeight) {
					var point = estimator.random(given);
					point.estimator = i;
					return point;
				}
			}
			throw new Error("No estimator selected!");
		},
		mean: function (given) {
			var result = [];
			given = given || [];
			for (var i = 0; i < this.varCount; i++) {
				if (given[i] == null) {
					given[i] = null;
					result[i] = 0;
				} else {
					result[i] = given[i];
				}
			}
			var weights = [];
			var totalWeight = 0;
			for (var i = 0; i < this.estimators.length; i++) {
				var estimator = this.estimators[i];
				var confidence = this.confidence[i];
				weights[i] = estimator.likelihood(given)*confidence;
				totalWeight += weights[i];
			}
			for (var i = 0; i < this.estimators.length; i++) {
				var estimator = this.estimators[i];
				var weight = weights[i]/Math.max(totalWeight, TINY_AMOUNT);
				if (weight == 0) { // No measurements
					continue;
				}
				for (var j = 0; j < result.length; j++) {
					if (given[j] == null) {
						var dist = estimator.collapse(j, given);
						var mean = dist.mean;
						result[j] += weight*mean;
						dist = estimator.collapse(j, given);
					}
				}
			}
			return result;
		},
		addConditionalObservation: function (given, vector, weight) {
			if (weight == undefined) {
				weight = 1;
			}
			var weights = [];
			var bestDistance2 = NaN;
			var bestEstimator = null;
			var totalWeight = 0;
			var totalConfidence = 0;
			for (var i = 0; i < this.estimators.length; i++) {
				var estimator = this.estimators[i];
				var confidence = this.confidence[i];
				totalConfidence += confidence;
				weights[i] = Math.max(estimator.likelihood(given)*confidence, TINY_AMOUNT);
				totalWeight += weights[i];
				var distance2 = 0;
				if (estimator.total == 0) {
					distance2 = Number.MAX_VALUE
				} else {
					var guess = estimator.random(given);
					for (var j = 0; j < given.length; j++) {
						distance2 += (vector[j] - guess[j])*(vector[j] - guess[j]);
					}
				}
				if (!(bestDistance2 < distance2)) {
					bestDistance2 = distance2;
					bestEstimator = estimator;
				}
			}
			var addedToEmpty = false;
			for (var i = 0; i < this.estimators.length; i++) {
				var estimator = this.estimators[i];
				var estimatorWeight = weights[i]/totalWeight;
				// Update confidence
				// TODO: the confidence update should be Bayesian-ish, not... whatever this is.
				if (estimator == bestEstimator) {
					this.confidence[i] += weight;//*(1 - estimatorWeight);
					estimator.addObservation(vector, weight);
				} else if (estimator.total == 0) {
					if (!addedToEmpty) {
						addedToEmpty = true;
						estimator.addObservation(vector, estimatorWeight*weight);
					}
				} else {
					//estimator.addObservation(vector, weight*estimatorWeight);
					this.confidence[i] -= weight*estimatorWeight;
					this.confidence[i] = Math.max(this.confidence[i], this.minimumConfidence);
				}
			}
			if (Math.random() < this.mutationProb) {
				// Mutation incorporating last observation
				this.killMutate([vector]);
			}
			console.log("Total confidence: " + totalConfidence);
			if (totalConfidence > this.confidenceLimit) {
				for (var i = 0; i < this.confidence.length; i++) {
					this.confidence[i] *= this.confidenceLimit/totalConfidence;
				}
			}
		}
	};

	api.GaussianEstimator = GaussianEstimator;
	api.Gaussian1D = Gaussian1D;
	api.MultiEstimator = MultiEstimator;
})((typeof module !== 'undefined' && typeof module.exports !== 'undefined') ? module.exports : (this.egc = {}));