import torch


def gram_matrix(features):
    b, c, h, w = features.shape
    features = features.view(b, c, h * w)
    Gramm = torch.matmul(features, features.transpose(1, 2))
    return Gramm.div(c * h * w)


def decode(action, canvas, decoder, num_strokes, num_stroke_params):
    canvas_size = canvas.shape[-1]
    action = action.view(-1, num_stroke_params)
    stroke = 1 - decoder(action[:, :10])
    stroke = stroke.view(-1, 1, canvas_size, canvas_size)
    color_stroke = stroke * action[:, -3:].view(-1, 3, 1, 1)
    stroke = stroke.view(-1, num_strokes, 1, canvas_size, canvas_size)
    color_stroke = color_stroke.view(-1, num_strokes, 3, canvas_size, canvas_size)
    for i in range(num_strokes):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas


def soft_update(target, source, polyak_factor):
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        target_p.data.copy_(target_p.data * polyak_factor + source_p.data * (1 - polyak_factor))


def hard_update(target, source):
    for target_m, source_m in zip(target.modules(), source.modules()):
        target_m._buffers = source_m._buffers.copy()
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        target_p.data.copy_(source_p.data)
