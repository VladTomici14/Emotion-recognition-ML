import matplotlib.pyplot as plt
import time
import os


def writeLogFile(date, start_time, end_time, accuracy, loss, auc, precision, val_accuracy, val_loss, val_auc,
                 val_precision, text):
    # ------- creating the log directory ------
    os.mkdir(f"models/{text}")

    # ---- writing the markdown file -----
    file = open(f"models/{text}/{text}.md", "w")
    file.write(f"# {text}.hdf5 - {date} \n")
    file.write(f"### Start time: {time.localtime(start_time)} \n")
    file.write(f"### End time: {time.localtime(end_time)} \n")

    file.write(f"## Results \n")
    file.write(f"### Time: {end_time - start_time} \n")
    file.write(f"#### Accuracy: {accuracy} \n")
    file.write(f"#### Loss: {loss} \n")
    file.write(f"#### Auc: {auc} \n")
    file.write(f"#### Precision: {precision} \n")

    file.write(f"## Validation results \n")
    file.write(f"#### Val_accuracy: {val_accuracy} \n")
    file.write(f"#### Val_loss: {val_loss} \n")
    file.write(f"#### Val_auc: {val_auc} \n")
    file.write(f"#### Val_precision: {val_precision} \n")

    # ----- adding the image to the md file ---------
    file.write(f"![Chart for results]({text}.png) \n")

    file.close()


def trainValidationPlot(accuracy, validation_accuracy,
                        loss, validation_loss,
                        auc, validation_auc,
                        precision, validation_precision, text):
    figure, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    figure.suptitle(f"Model's metrics visualization - models/{text}.hdf5")

    # --- histogram of accuracy ----
    ax1.set_title("History of accuracy")
    ax1.plot(range(1, len(validation_accuracy) + 1), validation_accuracy)
    ax1.plot(range(1, len(accuracy) + 1), accuracy)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend(['training', 'validation'])

    # --- histogram of loss ----
    ax2.set_title("History of loss")
    ax2.plot(range(1, len(validation_loss) + 1), validation_loss)
    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("loss")
    ax2.legend(['training', 'validation'])

    # --- histogram of auc ----
    ax3.set_title("History of auc")
    ax3.plot(range(1, len(validation_auc) + 1), validation_auc)
    ax3.plot(range(1, len(auc) + 1), auc)
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("auc")
    ax3.legend(['training', 'validation'])

    # --- histogram of precision ----
    ax4.set_title("History of precision")
    ax4.plot(range(1, len(validation_precision) + 1), validation_precision)
    ax4.plot(range(1, len(precision) + 1), precision)
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("precision")
    ax4.legend(['training', 'validation'])

    plt.savefig(f"models/{text}/{text}.png")
    print("Saved the plot !")
