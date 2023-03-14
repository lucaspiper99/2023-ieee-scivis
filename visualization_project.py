#!/usr/bin/env vtkpython
import sys
import random
import numpy as np
import pandas as pd
import vtk


random.seed(42)

WIDTH = 1000
HEIGHT = 600
SAMPLE_SIZE = 1000
SIMS = ['viz-no-network', 'viz-disable', 'viz-stimulus', 'viz-calcium']
NEURON_IDS = random.sample(range(50000), 50000)
STEP = 50


def getPoints(simIdx=-1):
    positionFiles = [f'SciVisContest23/{sim}/positions/rank_0_positions.txt' for sim in SIMS]
    points = []
    for i in range(4):
        pointLists = []
        points.append(vtk.vtkPoints())
        with open(positionFiles[i], 'r') as file:
            while True:
                line = file.readline()
                if not line or not line.strip():
                    break

                if line[0] == '#':
                    pass
                else:
                    content = line.split()
                    pointLists.append([float(x) for x in content[1:4]])
                    points[-1].InsertNextPoint(pointLists[-1])

            file.close()

    return points[simIdx] if -1 < simIdx < 4  else points

def getConnectionCells(connectionMap, ids):
    cells = vtk.vtkCellArray()
    for i in ids:
        if i in connectionMap.keys():
            for j in connectionMap[i]:
                polyLine = vtk.vtkLine()
                polyLine.GetPointIds().SetId(0, i)
                polyLine.GetPointIds().SetId(1, j)
                cells.InsertNextCell(polyLine)
    
    return cells


def getConnectionMap(step=STEP):
    connectionFiles = []
    for sim in SIMS:
        connectionFiles.append(
            [(f'SciVisContest23/{sim}/network/rank_0_step_{int(i*1e4)}_in_network.txt',
            f'SciVisContest23/{sim}/network/rank_0_step_{int(i*1e4)}_out_network.txt') for i in range(101)]
        )
    connectionMap = [{} for _ in SIMS]
    for i in range(4):
        with open(connectionFiles[i][step][0], 'r') as file:
            while True:
                line = file.readline()
                if not line or not line.strip():
                    break

                if line[0] == '#':
                    pass
                else:
                    content = line.split()
                    n1 = int(content[1]) - 1
                    n2 = int(content[3]) - 1
                    try:
                        connectionMap[i][n1].append(n2)
                    except KeyError:
                        connectionMap[i][n1] = [n2]
    file.close()

    return connectionMap


def getCalciumLevels():
    currentCalciumLevels = np.zeros((4, 100, 50_000))
    targetCalciumLevels = np.zeros((4, 100, 50_000))
    for i, sim in enumerate(SIMS):
        currentCalciumFiles = f'SciVisContest23/{sim}_current.csv'
        targetCalciumFiles = f'SciVisContest23/{sim}_target.csv'
        currentCalciumLevels[i][:][:] = pd.read_csv(currentCalciumFiles, header=None).to_numpy().T
        targetCalciumLevels[i][:][:] = pd.read_csv(targetCalciumFiles, header=None).to_numpy().T

    return currentCalciumLevels, targetCalciumLevels


def getSlider(renderWindowInteractor, renderer, title, init, max, x1, x2):
    slider_rep = vtk.vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(0)
    slider_rep.SetMaximumValue(max)
    slider_rep.SetValue(init)
    slider_rep.SetTitleText(title)
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(x1, .06)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(x2, .06)
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.3)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.3)
    slider_rep.SetTubeWidth(0.05)
    slider_rep.SetLabelFormat("%.0lf")
    slider_rep.SetTitleHeight(0.15)
    slider_rep.SetLabelHeight(0.15)
    slider = vtk.vtkSliderWidget()
    slider.SetInteractor(renderWindowInteractor)
    slider.SetAnimationModeToAnimate()
    slider.SetRepresentation(slider_rep)
    slider.KeyPressActivationOff()
    slider.SetAnimationModeToAnimate()
    slider.SetEnabled(True)
    slider.SetCurrentRenderer(renderer)

    return slider


def getScalarBar(mapper):
    scalarBarActor = vtk.vtkScalarBarActor()
    scalarBarActor.SetLookupTable(mapper.GetLookupTable())
    scalarBarActor.SetTitle('Calcium Level\n')
    scalarBarActor.UnconstrainedFontSizeOn()
    scalarBarActor.SetNumberOfLabels(5)
    scalarBarActor.SetMaximumWidthInPixels(int(np.ceil(WIDTH * .1)))
    scalarBarActor.SetMaximumHeightInPixels(int(np.ceil(HEIGHT * .9)))
    scalarBarActor.SetBarRatio(scalarBarActor.GetBarRatio() * 3.5)
    scalarBarActor.SetPosition(0.4, 0.1)

    return scalarBarActor


def getLUT():
    
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()

    colors = vtk.vtkNamedColors()
    p1 = [0.0] + list(colors.GetColor3d('MidnightBlue'))
    p2 = [0.5] + list(colors.GetColor3d('Gainsboro'))
    p3 = [1.0] + list(colors.GetColor3d('DarkOrange'))
    ctf.AddRGBPoint(*p1)
    ctf.AddRGBPoint(*p2)
    ctf.AddRGBPoint(*p3)

    table_size = 256
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(table_size)
    lut.Build()

    for i in range(0, table_size):
        rgba = list(ctf.GetColor(float(i) / table_size))
        rgba.append(1)
        lut.SetTableValue(i, rgba)

    return lut


def getCalciumLevelArray(calciumLevels, step):
    calciumLevelArray = vtk.vtkFloatArray()
    calciumLevelArray.SetName(f"Calcium_Levels")
    for j in calciumLevels[step]:
        calciumLevelArray.InsertNextValue(j)

    return calciumLevelArray


def getSources(connectionMap, calciumLevels, step=STEP, useSample=True, sampleSize=SAMPLE_SIZE):

    points = getPoints()
    ids = NEURON_IDS[:sampleSize] if useSample else NEURON_IDS[:]
    cells = [getConnectionCells(map, ids) for map in connectionMap]
    neuronPolyData = []
    connectionsPolyData = []
    glyphFilter = []
    for i in range(4):

        neuronPolyData.append(vtk.vtkPolyData())
        connectionsPolyData.append(vtk.vtkPolyData())

        neuronPolyData[-1].SetPoints(points[i])
        connectionsPolyData[-1].SetPoints(points[i])
        connectionsPolyData[-1].SetLines(cells[i])
        
        calciumLevelArray = getCalciumLevelArray(calciumLevels[i], step)
        neuronPolyData[-1].GetPointData().SetScalars(calciumLevelArray)

        glyphFilter.append(vtk.vtkVertexGlyphFilter())
        glyphFilter[-1].SetInputData(neuronPolyData[i])
        glyphFilter[-1].Update()

    return connectionsPolyData, glyphFilter


def getRewiring(currentMap, referenceMap):
    if not (type(referenceMap) == list and len(referenceMap) == 4):
        referenceMap = [referenceMap for _ in range(4)]

    maintained = [{} for _ in SIMS]
    created = [{} for _ in SIMS]
    deleted = [{} for _ in SIMS]

    for i in range(4):
        for n1 in currentMap[i]:
            if n1 not in referenceMap[i]:
                created[i][n1] = currentMap[i][n1]
            else:
                for n2 in currentMap[i][n1]:
                    if n2 in referenceMap[i][n1]:
                        try:
                            maintained[i][n1].append(n2)
                        except KeyError:
                            maintained[i][n1] = [n2]
                    else:
                        try:
                            created[i][n1].append(n2)
                        except KeyError:
                            created[i][n1] = [n2]

        for n1 in referenceMap[i]:
            if n1 not in currentMap[i]:
                deleted[i][n1] = referenceMap[i][n1]
            else:
                for n2 in referenceMap[i][n1]:
                    if n2 not in currentMap[i][n1]:
                        try:
                            deleted[i][n1].append(n2)
                        except KeyError:
                            deleted[i][n1] = [n2]

    return maintained, created, deleted


def getNormalVectors(polyData):
    normals = vtk.vtkPCANormalEstimation()
    normals.SetInputData(polyData)
    normals.SetSampleSize(50)
    normals.SetNormalOrientationToGraphTraversal()
    normals.Update()

    return normals


def getDiskSource():
    diskSource = vtk.vtkDiskSource()
    diskSource.SetInnerRadius(0.)
    diskSource.SetOuterRadius(2.5)
    diskSource.SetCircumferentialResolution(10)
    diskSource.SetNormal(1.,0.,0.)

    return diskSource


def task1():

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(WIDTH, HEIGHT)
    renderWindow.SetWindowName("SciVis Contest 2023")
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Viewport Ranges
    xMin = [  0, .45,   0, .45,  0, .9]
    xMax = [.45,  .9, .45,  .9,  1,  1]
    yMin = [.55, .55,  .1,  .1,  0, .1]
    yMax = [  1,   1, .55, .55, .1,  1]

    currentCalciumLevels, _ = getCalciumLevels()
    connectionMap = getConnectionMap()
    connectionsPolyData, glyphFilter = getSources(connectionMap, currentCalciumLevels)
    lut = getLUT()
    renderers = []
    neuronMappers = []
    connectionMappers = []

    for i in range(6):
        renderers.append(vtk.vtkRenderer())
        renderWindow.AddRenderer(renderers[-1])
        renderers[-1].SetViewport(xMin[i], yMin[i], xMax[i], yMax[i])

        if i < 4:
            if i == 0:
                camera = renderers[-1].GetActiveCamera()
            else:
                renderers[-1].SetActiveCamera(camera)

            neuronMappers.append(vtk.vtkPolyDataMapper())
            connectionMappers.append(vtk.vtkPolyDataMapper())

            neuronMappers[-1].SetInputConnection(glyphFilter[i].GetOutputPort())
            neuronMappers[-1].SetScalarRange(np.amin(currentCalciumLevels), np.amax(currentCalciumLevels))
            neuronMappers[-1].SetLookupTable(lut)
            connectionMappers[-1].SetInputData(connectionsPolyData[i])

            neuronActor = vtk.vtkActor()
            neuronActor.GetProperty().SetPointSize(7)
            neuronActor.GetProperty().SetRenderPointsAsSpheres(1)

            connectionActor = vtk.vtkActor()
            connectionActor.GetProperty().SetLineWidth(2)
            connectionActor.GetProperty().SetRenderLinesAsTubes(1)
            connectionActor.GetProperty().SetOpacity(0.8)

            textActor = vtk.vtkTextActor()
            textActor.SetInput(SIMS[i])
            txtprop = textActor.GetTextProperty()
            txtprop.SetFontFamilyToArial()
            txtprop.BoldOn()
            txtprop.SetFontSize(18)
            txtprop.SetBackgroundRGBA(0,0,0,.5)
            txtprop.SetColor(1, 1, 1)
            textActor.SetDisplayPosition(
                int(np.ceil(xMin[i]*WIDTH)) + 10,
                int(np.ceil(yMin[i]*HEIGHT)) + 10
                )
            neuronActor.SetMapper(neuronMappers[-1])
            connectionActor.SetMapper(connectionMappers[-1])
            renderers[-1].AddActor(neuronActor)
            renderers[-1].AddActor(connectionActor)
            renderers[-1].AddActor(textActor)

            renderers[-1].ResetCamera()

    def processEndInteractionEvent1(obj, event):
        global STEP
        value = int(round(obj.GetRepresentation().GetValue()))
        STEP = value
        connectionMap = getConnectionMap(step=value)
        polyData, glyphFilter = getSources(connectionMap, currentCalciumLevels, step=STEP, sampleSize=SAMPLE_SIZE)
        for i in range(4):
            neuronMappers[i].SetInputConnection(glyphFilter[i].GetOutputPort())
            connectionMappers[i].SetInputData(polyData[i])
        renderWindow.Render()

    def processEndInteractionEvent2(obj, event):
        global SAMPLE_SIZE
        value = int(round(obj.GetRepresentation().GetValue()))
        SAMPLE_SIZE = value
        polyData, _ = getSources(connectionMap, currentCalciumLevels, step=STEP, sampleSize=SAMPLE_SIZE)
        for i in range(4):
            connectionMappers[i].SetInputData(polyData[i])
        renderWindow.Render()
    
    slider1 = getSlider(renderWindowInteractor, renderers[4], "Simulation Step / 10^4", init=STEP, max=99, x1=.05, x2=.45)
    slider2 = getSlider(renderWindowInteractor, renderers[4],"Connection Sample Size", init=SAMPLE_SIZE, max=50_000, x1=.55, x2=.95)
    slider1.AddObserver("EndInteractionEvent", processEndInteractionEvent1)  # change value only when released
    slider2.AddObserver("EndInteractionEvent", processEndInteractionEvent2)  # change value only when released

    scalarBarActor = getScalarBar(neuronMappers[-1])
    renderers[5].AddActor(scalarBarActor)

    renderWindow.Render()
    renderWindowInteractor.Start()


def task2(simIdx):

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(WIDTH, HEIGHT)
    renderWindow.SetWindowName("SciVis Contest 2023")
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    connectionMap = getConnectionMap()

    calcLevels = list(getCalciumLevels())
    calcLevelArrays = [getCalciumLevelArray(calcLevels[0][simIdx], STEP), getCalciumLevelArray(calcLevels[1][simIdx], STEP)]
    points = [getPoints(simIdx), getPoints(simIdx)]
    polyData = [vtk.vtkPolyData(), vtk.vtkPolyData()]
    for i in range(2):
        polyData[i].SetPoints(points[i])
        polyData[i].GetPointData().SetScalars(calcLevelArrays[i])

    normals = [getNormalVectors(p) for p in polyData]
    source = getDiskSource()

    referenceMap = getConnectionMap(step=STEP-1)
    _, createdConMap, deletedConMap = getRewiring(connectionMap, referenceMap)
    createdConPolyData, _ = getSources(createdConMap, calcLevels[0], useSample=False)
    deletedConPolyData, _ = getSources(deletedConMap, calcLevels[0], useSample=False)

    xMin = [  0, .45,   0,  0, .9]
    xMax = [.45,  .9,  .9,  1,  1]
    yMin = [ .1,  .1, .55,  0, .1]
    yMax = [.55, .55,   1, .1,  1]
    lut = getLUT()
    renderers = []
    glyphs = []
    neuronMappers = []
    description = ['Current Calcium Level', 'Target Calcium Level', "Last Rewiring"]


    for i in range(5):
        renderers.append(vtk.vtkRenderer())
        renderWindow.AddRenderer(renderers[-1])
        renderers[-1].SetViewport(xMin[i], yMin[i], xMax[i], yMax[i])

        if i < 3:

            if i == 0:
                camera = renderers[-1].GetActiveCamera()
            else:
                renderers[-1].SetActiveCamera(camera)

            
            if i < 2:

                glyphs.append(vtk.vtkGlyph3D())
                glyphs[-1].SetInputConnection(normals[i].GetOutputPort())
                glyphs[-1].SetSourceConnection(source.GetOutputPort())
                glyphs[-1].SetColorModeToColorByScalar()
                glyphs[-1].SetVectorModeToUseNormal()
                glyphs[-1].ScalingOff()
                glyphs[-1].Update()


                neuronMappers.append(vtk.vtkPolyDataMapper())
                neuronMappers[-1].SetInputConnection(glyphs[-1].GetOutputPort())
                neuronMappers[-1].SetScalarRange(
                    np.amin(np.concatenate((calcLevels[0][simIdx], calcLevels[1][simIdx]))),
                    np.amax(np.concatenate((calcLevels[0][simIdx], calcLevels[1][simIdx])))
                    )
                neuronMappers[-1].SetLookupTable(lut)
                
                actor = vtk.vtkActor()
                actor.SetMapper(neuronMappers[-1])
                renderers[-1].AddActor(actor)

            else:

                createdConMappers = vtk.vtkPolyDataMapper()
                deletedConMappers = vtk.vtkPolyDataMapper()

                createdConMappers.SetInputData(createdConPolyData[simIdx])
                deletedConMappers.SetInputData(deletedConPolyData[simIdx])


                createdConActor = vtk.vtkActor()
                createdConActor.SetMapper(createdConMappers)
                createdConActor.GetProperty().SetLineWidth(2)
                createdConActor.GetProperty().SetRenderLinesAsTubes(1)
                createdConActor.GetProperty().SetColor(.09,.86,.15)  # green

                deletedConActor = vtk.vtkActor()
                deletedConActor.SetMapper(deletedConMappers)
                deletedConActor.GetProperty().SetLineWidth(2)
                deletedConActor.GetProperty().SetRenderLinesAsTubes(1)
                deletedConActor.GetProperty().SetColor(.87,.13,.06)  # red

                renderers[-1].AddActor(createdConActor)
                renderers[-1].AddActor(deletedConActor)

            textActor = vtk.vtkTextActor()
            textActor.SetInput(description[i])
            txtprop = textActor.GetTextProperty()
            txtprop.SetFontFamilyToArial()
            txtprop.BoldOn()
            txtprop.SetFontSize(14)
            txtprop.SetBackgroundRGBA(0,0,0,.5)
            txtprop.SetColor(1, 1, 1)
            textActor.SetDisplayPosition(
                int(np.ceil(xMin[i]*WIDTH)) + 10,
                int(np.ceil(yMin[i]*HEIGHT)) + 10
                )
            renderers[-1].AddActor(textActor)

            renderers[-1].ResetCamera()


    def processEndInteractionEvent(obj, event):
        value = int(round(obj.GetRepresentation().GetValue()))
        
        calcLevelArrays = [
            getCalciumLevelArray(calcLevels[0][simIdx], value),
            getCalciumLevelArray(calcLevels[1][simIdx], value)
            ]
        polyData = [vtk.vtkPolyData(), vtk.vtkPolyData()]
        normals = []
        for i in range(2):
            polyData[i].SetPoints(points[i])
            polyData[i].GetPointData().SetScalars(calcLevelArrays[i])
            polyData[i].Modified()
            normals.append(getNormalVectors(polyData[i]))
            glyphs[i].SetInputData(normals[i].GetOutput())
            neuronMappers[i].Update()

            value = 1 if value == 0 else value
            connectionMap = getConnectionMap(step=value)
            referenceMap = getConnectionMap(step=value-1)
            _, createdConMap, deletedConMap = getRewiring(connectionMap, referenceMap)
            createdConPolyData, _ = getSources(createdConMap, calcLevels[0], useSample=False)
            deletedConPolyData, _ = getSources(deletedConMap, calcLevels[0], useSample=False)
            createdConMappers.SetInputData(createdConPolyData[simIdx])
            deletedConMappers.SetInputData(deletedConPolyData[simIdx])
        
            renderWindow.Render()
    
    slider = getSlider(renderWindowInteractor, renderers[3], "Simulation Step / 10^4", init=50, max=99, x1=.15, x2=.85)
    slider.AddObserver("EndInteractionEvent", processEndInteractionEvent)  # change value only when released

    scalarBarActor = getScalarBar(neuronMappers[-1])
    renderers[4].AddActor(scalarBarActor)

    renderWindow.Render()
    renderWindowInteractor.Start()


def main(argv):

    argsError = 'arguments:\ntask\t\tproject task\nsimulation\tsimulation index (only used in task 2)'
    optError = 'options:\ntask\t\t1, 2\nsimulation\t1, 2, 3, 4'
    try:
        cond1 = len(argv) < 2
        cond2 = int(argv[1]) not in [1, 2]
        cond3 = int(argv[1]) == 2 and len(argv) == 2
        cond4 = int(argv[1]) == 2 and not argv[2].isnumeric()
        cond5 = int(argv[1]) == 2 and int(argv[2]) not in [1, 2, 3, 4]
    except IndexError:
        sys.stderr.write("Usage: %s <task> <simulation>\n\n%s\n\n%s" % (argv[0], argsError, optError))
        return 1

    if cond1 or cond2 or cond3 or cond4 or cond5:
        sys.stderr.write("Usage: %s <task> <simulation>\n\n%s\n\n%s" % (argv[0], argsError, optError))
        return 1

    if int(argv[1]) == 1:
        print('Running:\ntask\t\t1')
        task1()
    if int(argv[1]) == 2:
        simIdx = int(argv[2]) - 1
        print("Running:\ntask\t\t2\nsimulation\t%s [%s]" % (argv[2], SIMS[simIdx]))
        task2(simIdx)


if __name__ == '__main__':
    main(sys.argv)
