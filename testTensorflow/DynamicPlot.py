import matplotlib.pyplot as plt

class DynamicPlot:

    def on_launch(self,title):
        plt.ion();
        self.figure, self.ax = plt.subplots()
        self.figure.suptitle(title);
        self.ax.set_autoscaley_on(True)
        self.ax.grid()

    def setXaxisTitle(self,title):
        plt.xlabel(title);
    def setYaxisTitle(self,title):
        plt.ylabel(title);
    def setXYaxisTitle(self,titleX,titleY):
        plt.xlabel(titleX);
        plt.ylabel(titleY);

    def requestLines(self,title,info='-'):
        lines, = self.ax.plot([],[],info,label = title);
        return lines

    def on_running(self,lines, xdata, ydata):
        #Update data (with the new _and_ the old points)
        lines.set_xdata(xdata)
        lines.set_ydata(ydata)

    def draw(self):
        plt.legend();
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def done(self):
        plt.ioff();
    #Example
#    def __call__(self):
#        import numpy as np
#        import time
#        self.on_launch()
#        xdata = []
#        ydata = []
#        for x in np.arange(0,10,0.5):
#            xdata.append(x)
#            ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
#            self.on_running(xdata, ydata)
            #time.sleep(1)
#       return xdata, ydata

