# Multi-class quantification of Frog Embryos-Devision Team-Spring 2024

In Spring 2024, Dr. Peter Wolenski and the Devision team aimed to develop a program that uses multi-class prediction to count frog embryos at different developmental stages within a Petri dish, specifically focusing on the Cleavage stage (NF 1 to 6) of Xenopus laevis. This project was done in collaboration with AGGRC and the Marine Biological Laboratory at the University of Chicago. The embryos undergo cell divisions without growing in size, maintaining approximately 1.5 mm diameter.

The data, including views from animal, vegetal, and lateral perspectives, were annotated using Fiji and LabKit, categorizing viable and non-viable embryos by assigning distinct colors to each stage. Stardist, a neural network based on U-net, was utilized to predict the viable and non-viable stage of embryos.

The training process, involving an 80-20 split of training and testing datasets, showed significant improvement in model performance, indicated by decreasing distance and probability loss over epochs.

The results demonstrated high F1 scores, accuracy, recall, and precision. Future plans for Summer 2024 include expanding the classification algorithm to encompass all developmental stages and developing a user-friendly GUI application for laboratory use.

![Image 0](images/eggs.png)

# Research done by the Spring 2024 DeVision Team
## CEO
Dr. Peter Wolenski wolenski@math.lsu.edu


## Assistant Professor
Dr. Nadejda Drenska ndrenska@lsu.edu


## Graduate Team Managers:
Iswarya Sitiraju, Gowri Priya Sunkara


## Mathematics Graduate Students:
Oluwaferanmi D. Agbolade, Kenneth Betterman, Christian Ennis


## Undergraduate Students:
Erica Clement, Ravyn Johnlouis, Han Nguyen, Yahreia Peeler, Jamar K. Whitfield

## AGGRC

Director Terrance Tiersch

Assistant Director- Prof. Yue Liu

Post Doc Dr. Jack Koch

https://aggrc.com/

## The University of Chicago - Marine Biological Laboratory

Research Assistant: Carl Anderson

https://new-www.mbl.edu/
<br>
<br>

<img src="images/mcclogo.gif" alt="Image 2" width="100">
LSU Math Consultation Clinic:<br>
https://www.math.lsu.edu/courses/capstone_course
<br>
<br>

<img src="images/lsulogo.png" alt="Image 1" width="250">
https://lsu.edu/
