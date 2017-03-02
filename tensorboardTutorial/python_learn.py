import matplotlib.pyplot as plt

class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 10

    def on_launch(self):
        plt.ion();
        #Set up plot
        self.figure, self.ax = plt.subplots()
        #self.axh = plt.axvline()
       # self.lines, = self.ax.plot([],[], 'o')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        #self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()

    def requestLines(self,title,info='-'):
        lines, = self.ax.plot([],[],info,label = title);
        return lines

    def addLine(self,lines, xdata, ydata):
        #Update data (with the new _and_ the old points)
        lines.set_xdata(xdata)
        lines.set_ydata(ydata)

    def addVertial(self,xdata):
        plt.axvline(xdata)


    def draw(self):
        plt.legend()
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()

        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    def done(self):
        plt.ioff();
    #Example
    def __call__(self):
        import numpy as np

        self.on_launch()
        xdata = []
        ydata = []

        xdata2 = []
        ydata2 = []

        lines= self.requestLines('hey',info='r-o');
        lines2 = self.requestLines('yo',info='b-');
        for x in np.arange(0,10,0.5):
            y1= np.exp(-x**2)+10*np.exp(-(x-7)**2)
            y2 = np.exp(-x ** 3) + 4 * np.exp(-(x - 2) ** 2)




            self.addLine(lines,x, [-0,1])
            self.addLine(lines2, x, y2)

            self.draw()

        self.done();
        plt.show()


d = DynamicUpdate()
d()
