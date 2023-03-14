# Visualization Project

## Description

The goal of this project is to ...

## How to Run

In order to access the visualization, it's necessary to have Python installed with all the modules specified in the [requirements file](requirements.txt). Besides these modules, it's necessary to have the [VTK Python binding](https://vtk.org/download/) installed. As for the data, it must be set in the following way:

- Create a `SciVis2023` folder
- Download [the original data set](https://rwth-aachen.sciebo.de/s/KNTo1vgT0JZyGJx) and put all the files in the `SciVis2023` folder
- Download [this extra file](https://drive.google.com/file/d/1jFK71ZevDojzTRM2krZ1VVCTflAWSTr8/view?usp=sharing), unzip it inside the `SciVisContest23` folder

Afterwards, the visualization can be launched by running `visualization_project.py <task> <simulation>`, where:

### Arguments

- `task`: project task
- `simulation`: simulation index (only used in task 2)

### Options

- `task`: 1, 2
- `simulation`: 1, 2, 3, 4
