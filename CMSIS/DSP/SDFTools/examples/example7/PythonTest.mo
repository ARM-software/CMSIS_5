model PythonTest
  ARM.Models.VHT vht(launchVHT = false, samplingFrequency = 16)  annotation(
    Placement(visible = true, transformation(origin = {-12, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.TransferFunction transferFunction(a = {1.0, 942.317, 3.94761e7}, b = {1, 1675.7, 3.94761e7})  annotation(
    Placement(visible = true, transformation(origin = {32, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay(delayTime = 0.1)  annotation(
    Placement(visible = true, transformation(origin = {68, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Add add annotation(
    Placement(visible = true, transformation(origin = {-36, -28}, extent = {{10, -10}, {-10, 10}}, rotation = -90)));
  Modelica.Blocks.Math.Gain gain(k = 0.8)  annotation(
    Placement(visible = true, transformation(origin = {102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Noise.TruncatedNormalNoise noise(samplePeriod = 0.0000625, useAutomaticLocalSeed = true, useGlobalSeed = true, y_max = 0.1, y_min = -0.1) annotation(
    Placement(visible = true, transformation(origin = {-82, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  inner Modelica.Blocks.Noise.GlobalSeed globalSeed annotation(
    Placement(visible = true, transformation(origin = {-86, -28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  ARM.Sound.WaveOutput waveOutput annotation(
    Placement(visible = true, transformation(origin = {24, -32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(vht.y, transferFunction.u) annotation(
    Line(points = {{0, 0}, {20, 0}}, color = {0, 0, 127}));
  connect(transferFunction.y, fixedDelay.u) annotation(
    Line(points = {{44, 0}, {56, 0}}, color = {0, 0, 127}));
  connect(add.y, vht.x) annotation(
    Line(points = {{-36, -16}, {-36, 0}, {-24, 0}}, color = {0, 0, 127}));
  connect(fixedDelay.y, gain.u) annotation(
    Line(points = {{80, 0}, {90, 0}}, color = {0, 0, 127}));
  connect(gain.y, add.u1) annotation(
    Line(points = {{114, 0}, {132, 0}, {132, -48}, {-30, -48}, {-30, -40}}, color = {0, 0, 127}));
  connect(noise.y, add.u2) annotation(
    Line(points = {{-70, 8}, {-54, 8}, {-54, -46}, {-42, -46}, {-42, -40}}, color = {0, 0, 127}));
  connect(vht.y, waveOutput.x) annotation(
    Line(points = {{0, 0}, {6, 0}, {6, -32}, {12, -32}}, color = {0, 0, 127}));
  annotation(
    Diagram(coordinateSystem(extent = {{-200, -200}, {200, 200}})),
    uses(ARM(version = "0.1.0"), Modelica(version = "4.0.0")));
end PythonTest;
