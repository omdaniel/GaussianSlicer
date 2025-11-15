// ContentView.swift
import SwiftUI
import MetalKit
import AppKit

struct ContentView: View {
    private let labelWidth: CGFloat = 200
    @State private var planeOffset: Float
    @StateObject private var settings: AppSettings
    @StateObject private var renderer: MetalRenderer
    @State private var exportNormalized: Bool = false
    
    init() {
        let runtimeConfig = RuntimeConfig.shared
        let appSettings = AppSettings(config: runtimeConfig)
        let rendererSettings = appSettings.makeRendererSettings()
        let initialOffset = max(rendererSettings.gridMin, min(0.0, rendererSettings.gridMax))
        _planeOffset = State(initialValue: initialOffset)
        _settings = StateObject(wrappedValue: appSettings)
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device initialization failed unexpectedly.")
        }
        _renderer = StateObject(wrappedValue: MetalRenderer(device: device, settings: rendererSettings))
    }

    var body: some View {
        HStack(spacing: 16) {
            mainContent
            Divider()
            sidebar
        }
        .padding()
    }
    
    private var header: some View {
        HStack {
            Text("3D GMM Slicing (Metal Accelerated)")
                .font(.headline)
            Spacer()
            Text("N=\(renderer.numDistributions) | Grid=\(renderer.gridResolution)x\(renderer.gridResolution)")
                .font(.subheadline)
            Text(String(format: "Frame Time: %.2f ms", renderer.frameTime))
                .font(.subheadline)
                .monospacedDigit()
                .padding(5)
                .background(Color.secondary.opacity(0.2))
                .cornerRadius(5)
        }
        .padding(.horizontal)
        .padding(.top, 10)
    }
    
    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            ScrollView {
                settingsPanelContent
                    .padding(.vertical, 4)
            }
            .frame(maxHeight: .infinity, alignment: .top)
            
            Spacer(minLength: 0)
            
            applyControls
        }
        .frame(width: 480)
    }
    
    private var mainContent: some View {
        VStack(spacing: 12) {
            header
            
            MetalView(renderer: renderer)
                .frame(minWidth: 512, maxWidth: .infinity, minHeight: 512, maxHeight: .infinity)
                .aspectRatio(1.0, contentMode: .fit)
                .border(Color.gray, width: 1)
                .onAppear {
                    renderer.currentOffset = planeOffset
                }
                .allowsHitTesting(false)
                .focusable(false)
                .layoutPriority(1)
                .overlay(alignment: .topLeading) {
                    OrientationGizmo(renderer: renderer)
                        .frame(width: 128, height: 128)
                        .padding(8)
                }
            
            Spacer(minLength: 0)
            
            offsetControl
        }
        .frame(minWidth: 600, maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
    
    private var offsetControl: some View {
        HStack {
            Text(String(format: "Offset: %.2f", planeOffset))
                .frame(width: 110, alignment: .leading)
            Slider(value: $planeOffset, in: renderer.gridMin...renderer.gridMax)
                .onChange(of: planeOffset) { newValue in
                    renderer.currentOffset = newValue
                }
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }
    
    private var settingsPanelContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            GroupBox("Data Generation") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        label("Distributions")
                        EditableTextField(text: $settings.numDistributionsText)
                            .frame(width: 120)
                    }
                    HStack {
                        label("Mean StdDev")
                        EditableTextField(text: $settings.meanStdDevText)
                            .frame(width: 120)
                    }
                    HStack {
                        label("Covariance Scale")
                        EditableTextField(text: $settings.covarianceScaleText)
                            .frame(width: 120)
                    }
                    HStack {
                        label("Random Seed")
                        EditableTextField(text: $settings.seedString)
                            .frame(width: 140)
                    }
                    HStack {
                        label("Plane Normal")
                        HStack(spacing: 8) {
                            EditableTextField(text: $settings.planeNormalXText)
                                .frame(width: 70)
                            EditableTextField(text: $settings.planeNormalYText)
                                .frame(width: 70)
                            EditableTextField(text: $settings.planeNormalZText)
                                .frame(width: 70)
                        }
                    }
                }
            }
            
            GroupBox("Grid") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        label("Resolution")
                        EditableTextField(text: $settings.gridResolutionText)
                            .frame(width: 120)
                    }
                    HStack {
                        label("Grid Min")
                        EditableTextField(text: $settings.gridMinText)
                            .frame(width: 120)
                    }
                    HStack {
                        label("Grid Max")
                        EditableTextField(text: $settings.gridMaxText)
                            .frame(width: 120)
                    }
                }
            }
            
            GroupBox("Visualization") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        label("Colormap")
                        Picker("", selection: $settings.colormap) {
                            ForEach(ColormapOption.allCases, id: \.self) { option in
                                Text(displayName(for: option)).tag(option)
                            }
                        }
                        .pickerStyle(.menu)
                        .frame(width: 160)
                    }
                    Toggle("Invert Colormap", isOn: $settings.invertColormap)
                        .toggleStyle(.switch)
                        .padding(.leading, labelWidth)
                    Toggle("Logarithmic Scaling", isOn: $settings.useLogScale)
                        .toggleStyle(.switch)
                        .padding(.leading, labelWidth)
                    HStack {
                        label("Density Min")
                        EditableTextField(text: $settings.densityMinText)
                            .frame(width: 140)
                    }
                    HStack {
                        label("Density Max")
                        EditableTextField(text: $settings.densityMaxText)
                            .frame(width: 140)
                    }
                    HStack {
                        label("Color Levels")
                        Stepper(value: $settings.colorLevels, in: 0...256, step: 1) {
                            Text(settings.colorLevels == 0 ? "Continuous" : "\(settings.colorLevels)")
                        }
                    }
                    HStack {
                        label("Outline Width")
                        EditableTextField(text: $settings.outlineWidthText)
                            .frame(width: 120)
                    }
                    Toggle("Nearest Neighbor Filter", isOn: $settings.useNearestFilter)
                        .toggleStyle(.switch)
                        .padding(.leading, labelWidth)
                }
            }
        }
    }
    
    private var applyControls: some View {
        HStack {
            Spacer()
            Toggle("Export normalized (log, 0–1)", isOn: $exportNormalized)
                .toggleStyle(.checkbox)
                .help("Write normalized log-scaled values to [0,1] using current density min/max and invert options.")
                .padding(.trailing, 8)
            Button("Apply Settings") {
                let sanitized = settings.makeRendererSettings()
                settings.sync(from: sanitized)
                planeOffset = max(min(planeOffset, sanitized.gridMax), sanitized.gridMin)
                renderer.apply(settings: sanitized)
                renderer.currentOffset = planeOffset
            }
            .buttonStyle(.borderedProminent)

            Button("Export Volume…") {
                if let window = NSApp.keyWindow ?? NSApp.mainWindow {
                    VolumeExporter.exportVolume(renderer: renderer, parentWindow: window, normalizedLog01: exportNormalized)
                } else {
                    VolumeExporter.exportVolume(renderer: renderer, parentWindow: nil, normalizedLog01: exportNormalized)
                }
            }
            .disabled(renderer.isExporting)

            Button("Export OpenVDB…") {
                if let window = NSApp.keyWindow ?? NSApp.mainWindow {
                    VolumeExporter.exportOpenVDB(renderer: renderer, parentWindow: window, normalizedLog01: exportNormalized)
                } else {
                    VolumeExporter.exportOpenVDB(renderer: renderer, parentWindow: nil, normalizedLog01: exportNormalized)
                }
            }
            .disabled(renderer.isExporting)
        }
        .padding(.top, 4)
    }
    
    @ViewBuilder
    private func label(_ text: String) -> some View {
        Text(text)
            .frame(width: labelWidth, alignment: .leading)
    }
    
    private func displayName(for option: ColormapOption) -> String {
        let raw = String(describing: option)
        let spaced = raw.replacingOccurrences(of: "([a-z0-9])([A-Z])",
                                              with: "$1 $2",
                                              options: .regularExpression)
        return spaced.capitalized
    }
}

private struct EditableTextField: NSViewRepresentable {
    @Binding var text: String

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text)
    }

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField(string: text)
        field.isEditable = true
        field.isBezeled = true
        field.bezelStyle = .roundedBezel
        field.drawsBackground = true
        field.backgroundColor = .windowBackgroundColor
        field.focusRingType = .default
        field.font = NSFont.monospacedDigitSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
        field.delegate = context.coordinator
        return field
    }

    func updateNSView(_ nsView: NSTextField, context: Context) {
        if nsView.stringValue != text {
            nsView.stringValue = text
        }
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        var text: Binding<String>

        init(text: Binding<String>) {
            self.text = text
        }

        func controlTextDidChange(_ obj: Notification) {
            guard let field = obj.object as? NSTextField else { return }
            text.wrappedValue = field.stringValue
        }
    }
}

// Helper view to bridge MTKView into SwiftUI for macOS
struct MetalView: NSViewRepresentable {
    @ObservedObject var renderer: MetalRenderer
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = NonFocusableMTKView()
        mtkView.delegate = renderer
        mtkView.device = renderer.device
        mtkView.isPaused = false
        mtkView.enableSetNeedsDisplay = false
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {}
}

// MARK: - Orientation Gizmo (SwiftUI overlay)
private struct OrientationGizmo: View {
    @ObservedObject var renderer: MetalRenderer

    private func axis2D(_ a: SIMD3<Float>) -> SIMD2<Float> {
        let d = SIMD2<Float>(a.x, a.y)
        let len = simd_length(d)
        if len < 1e-6 { return SIMD2<Float>(0, -1) } // default up if pointing mostly in Z
        return d / len
    }

    private func color(_ a: SIMD3<Float>, base: NSColor) -> Color {
        // Dim if pointing away (negative Z)
        let facing = max(0.0, CGFloat(a.z))
        let c = base.usingColorSpace(.deviceRGB) ?? base
        return Color(.sRGB, red: c.redComponent, green: c.greenComponent, blue: c.blueComponent, opacity: 0.5 + 0.5 * facing)
    }

    var body: some View {
        Canvas { ctx, size in
            let center = CGPoint(x: 24, y: size.height - 24) // top-left with padding inside
            let arm: CGFloat = 40
            let circleR: CGFloat = 12

            let axes = renderer.axesUVN
            let u2 = axis2D(axes.u)
            let v2 = axis2D(axes.v)
            let n2 = axis2D(axes.n)

            // Lines
            func drawAxis(_ axis3: SIMD3<Float>, dir: SIMD2<Float>, col: NSColor) {
                let end = CGPoint(x: center.x + CGFloat(dir.x) * arm,
                                  y: center.y + CGFloat(dir.y) * arm)
                var p = Path()
                p.move(to: center)
                p.addLine(to: end)
                ctx.stroke(p, with: .color(color(axis3, base: col)), lineWidth: 3)
                let circle = Path(ellipseIn: CGRect(x: end.x - circleR, y: end.y - circleR, width: circleR * 2, height: circleR * 2))
                ctx.fill(circle, with: .color(color(axis3, base: col)))
            }

            drawAxis(axes.u, dir: u2, col: .systemRed)
            drawAxis(axes.v, dir: v2, col: .systemGreen)
            drawAxis(axes.n, dir: n2, col: .systemBlue)

            // Labels
            func drawLabel(_ dir: SIMD2<Float>, text: String) {
                let end = CGPoint(x: center.x + CGFloat(dir.x) * arm,
                                  y: center.y + CGFloat(dir.y) * arm)
                let resolved = ctx.resolve(Text(text).font(.system(size: 11, weight: .semibold)).foregroundColor(.white))
                ctx.draw(resolved, at: CGPoint(x: end.x, y: end.y), anchor: .center)
            }

            drawLabel(u2, text: "X")
            drawLabel(v2, text: "Y")
            drawLabel(n2, text: "Z")
        }
    }
}

private final class NonFocusableMTKView: MTKView {
    override var acceptsFirstResponder: Bool { false }

    override func becomeFirstResponder() -> Bool { false }
}
