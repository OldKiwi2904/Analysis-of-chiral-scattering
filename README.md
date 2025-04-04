# Analysis-of-chiral-scattering
We are considering scattering spectra of various optically chiral helices of nm size.

The task is to find out how do the local extrema (maxima and minima) evolve moving from one spectrum to another with the change of the sample rotation angle. By date the code allows for finding the connection branches thru the manifold of given spectra, related to one considered helix:
1) Find local maxima of the experimentally derived spectra
2) Filter them and group closely located extrema
3) Search paths connecting the points of the neighboring spectra using one of the two metrics:
   a) distance on x-y plane
   b) distance in 3d space

The presented code has several limitations:
1) It can not allow for searching paths that have shorter length the number of spectra in a set. Thus, no every sequenced evolution of peaks can be detected precisely due to different number of local extrema on each spectrum
2) Sometimes when the extrema of the neighboring spectra are situated not close enough to each other the algorithm makes a mistake considering them as sequenced. We may apply filter via a certain quartile to avoid a miss
3) The code concentrates only around choosing from local maxima but not minima

Suggestions to apply machine learning:
1) Predict angular evolution of spectra
2) Interpolate the missing extrema on the way of found path if the surrounding spectra meet the mentioned condition
