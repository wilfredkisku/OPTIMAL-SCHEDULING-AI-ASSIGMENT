# OptimalScheduling-Artificial-Intelligence
Local Search for scheduling of presentations and obtaining a optimal schedule
## (A.1) Problem Statement:


To search for an optimal solution for scheduling N presentation in T time slots, each having P presentations and divided into S parallel sessions. The Measure of goodness for a praticular solution or scheme is given by a function G(sh).

## (B.1) Local Searh Approach and Heurisitic:


The idea the I have tried to implement are basically two approaches (1) Simulated Annealing and (2) Hill Climbing. The other idea to remove random assignment of state for calculation of the goodness measure is the use of a simple heuristic of evaluation of slots of presentation that are similar to each other.

There are two local search that have been implemented together so as to achieve the best schedule from both of the functions and then take the schedule with the maximum goodness function. There were other implementational idea that we experimented with but it they included obtaining the maximizing schedule by iterating over the local search algorithms and then picking the best schedule from it.

## (B.2) Simulated Annealing:

The algorithm is feed in with a randomly intialized state from the state space (randomly generated). The other values of importance in the algorithm is Temperature (T = 10.0), which is decreased at a rate of 0.95. The temperature is reduced which would decide the probability of selecting the bad moves, with lower values of T the probability of selection of bad schedule selection decreases exponentially. The other factor to be measured is del_GM which is the difference between the goodness measure of the newly generated schedule and the goodness measure of the current schedule. If the goodness mesure is positive then [Sched_Curr := Sched_Next] else [with probability of EXP(del_GM/T) assign Sched_Curr := Sched_Next].

## (B.3) Hill Climbing:

In the hill climbing approach we have just selected the best schedule that is better that the previous schedule, iterating for 1000 times.

## (B.4) Heuristic:

The heuristic that we have attempted on is to generate a schedule that tries to find the minimum distances between randomly selected presentations and arranging them into a slot, thus these slots would have the minimum distances or maximum similarity values.

## (C.1) Solution Details:


	Class		: assignment
	Subroutines	: goodnessMeasure
			  simulatedAnnealing
			  hillClimbing
			  generatedState, generatedState_n
			  outputResult

	variables	: similarity_matrix
			  distance_matrix
			  P, S, T and Z
	  
## (D.1) Executing Details:

python P19EE003_P19EE001_MT19AI015.py input_sample.txt output.txt

** Also the zip file can be executed through the command line by changing the souce filename to "__main__.py".

Status: Completed
