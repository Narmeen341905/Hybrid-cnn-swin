# ===============================
# FILE: src/training/train_fdcl.py
# ===============================
import os
import argparse
import tensorflow as tf
from tensorflow.keras import optimizers as opt

from src.datasets.loader import load_dataset_roots
from src.models.tailored_cnn import build_tailored_cnn
from src.models.swin_transformer import build_swin_backbone
from src.models.cross_fusion_block import cross_fusion_block
from src.models.heads import classification_head
from src.models.fdcl_losses import fdcl_loss


def build_model(input_shape=(256,256,3), n_classes=10,
                cnn_channels=(32,64,128), swin_embed=48, window=8):
    inp = tf.keras.Input(shape=input_shape)
    cnn = build_tailored_cnn(inp, channels=cnn_channels)
    swin = build_swin_backbone(inp, embed_dim=swin_embed, window=window)
    a, b, fused = cross_fusion_block(cnn, swin, out_channels=None)
    # Option: another small fusion of a and b
    fused2 = tf.keras.layers.Concatenate()([a, b])
    fused2 = tf.keras.layers.Conv2D(fused.shape[-1], 1, padding='same')(fused2)
    logits = classification_head(fused2, n_classes)
    model = tf.keras.Model(inp, logits, name="HybridCNN_Swin_FDCL")
    # expose an intermediate for FDCL
    model._fdcl_tensor = fused2
    return model


def main(args):
    data = load_dataset_roots(args.data_root, img_size=(args.img_h, args.img_w), batch=args.batch)
    n_classes = data["n_classes"]

    model = build_model(input_shape=(args.img_h, args.img_w, 3), n_classes=n_classes,
                        cnn_channels=(32,64,128), swin_embed=48, window=8)

    # Optimizer
    optimizer = opt.SGD(learning_rate=args.lr, momentum=args.momentum, nesterov=False)

    # Losses
    ce = tf.keras.losses.SparseCategoricalCrossentropy()

    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            ce_loss = ce(y, logits)
            total_loss = ce_loss
            if args.warmup_epochs <= 0 or tf.cast(model._train_counter, tf.float32) >= args.warmup_epochs:
                # FDCL after warmup
                fdcl = fdcl_loss(model._fdcl_tensor, proj_dim=args.proj_dim, temperature=args.temperature, cutoff=args.cutoff)
                total_loss = ce_loss + args.lambda_ccl * fdcl
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc.update_state(y, logits)
        return ce_loss, total_loss

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        val_acc.update_state(y, logits)

    best_val = 0.0
    model._train_counter = tf.Variable(0, dtype=tf.int32)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        # Train
        for step, (x, y) in enumerate(data["train"]):
            ce_loss, total_loss = train_step(x, y)
        model._train_counter.assign_add(1)
        # Validate
        for x, y in data["val"]:
            val_step(x, y)
        print(f"  acc: {train_acc.result().numpy():.4f} | val_acc: {val_acc.result().numpy():.4f}")
        if val_acc.result().numpy() > best_val:
            best_val = val_acc.result().numpy()
            os.makedirs(args.outdir, exist_ok=True)
            model.save(os.path.join(args.outdir, "best_model"))
        train_acc.reset_states(); val_acc.reset_states()

    # Final test
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in data["test"]:
        logits = model(x, training=False)
        test_acc.update_state(y, logits)
    print(f"Test Accuracy: {test_acc.result().numpy():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root folder with class subfolders")
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.00143)
    parser.add_argument("--momentum", type=float, default=0.0912)
    parser.add_argument("--lambda_ccl", type=float, default=0.5)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="experiments/ckpts")
    args = parser.parse_args()
    main(args)
