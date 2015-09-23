//Michael Skirpan - Fast Forward Labs
//Visualization of Simple Feed-forward Neural Network Using Backpropagation Algorithm
//Many thanks to:
//Daniel Shiffman - http://natureofcode.com - for education and example code
//P5.js folks - http://p5js.org - for example code
//Michael Nielson - http://neuralnetworksanddeeplearning.com/ - for lucid explanations of algorithms
//Andrej Karpathy - https://cs231n.github.io - for great attention to backpropagtion 

var myNetwork;
var batchIn;

function sigmoid(z){
    return 1.0/(1.0+Math.exp(-z));
}

function sigmoid_prime(z){
    return sigmoid(z)*(1-sigmoid(z));
}

// ############
// Neuron Object
// #############

var Neuron = function (location, bias, value) {
    //Set location, any initial value (for inputs), bias
    this.location = location || createVector(location[0], location[1]);
    this.connections = [];
    this.backconnects = [];
    this.val = value || 0;
    this.bias = bias || random(-1, 1);
    this.acc = 0;
    this.incoming = 0;
    this.sig;
    this.learningRate = .1;
    this.batch = false;
    //For backprop stuff


    //physical size
    this.r = 32;

    
    this.addConnection = function(c) {
        this.connections.push(c);
        c.b.incoming++;
    }  

    this.backConnect = function(c) {
        this.backconnects.push(c);
    }
  
  // Receive an input
  this.feedforward = function(input) {
    // Accumulate it
    this.val += input;
    this.acc++;

    // Activate it?
    if (this.acc == this.incoming) {
      this.val += this.bias;
      this.sig = sigmoid(this.val);
      this.acc = 0;
      if (this.sig > 0) {
        this.fire();
        }
      this.val = 0;
    } 
  }

  //Backpropagate
  this.backprop = function(grad) {
    //takes in the calculated gradiant leading to this node
    //derivative of sigmoid
    dsig = ((1 - this.sig) * this.sig) * grad;

    //update bias
    this.bias += (-this.learningRate * dsig)

    //shrink for backprop
    this.r = 12;

    //send update for weight
    if (this.batch){
        for (bc in this.backconnects) {
            this.backconnects[bc].batch = true;
            this.backconnects[bc].backprop(dsig);
        }
    } else {
        for (bc in this.backconnects) {
            this.backconnects[bc].backprop(dsig);
        }
    }
  }
  
  this.fire = function() {
    //make it bigger
    this.r = 64;   

    
    // We send the output through all connections
    if (this.batch) {
        for (c in this.connections) {
            this.connections[c].batch = true;
           this.connections[c].feedforward(this.val);
        } 
    } else {
        for (c in this.connections){
            this.connections[c].feedforward(this.val);
        }
    }
  }

  this.backDisplay = function() {
    stroke(0);
    strokeWeight(1);
    push();
        strokeWeight(3);
        fill('rgba(127,47,155, .2)');
        ellipse(this.location.x, this.location.y, this.r, this.r);
    pop();
    if (this.backconnects.length != 0) {
        push();
            fill(0, 102, 153);
            text("b= "+Math.round(this.bias * 10000) / 10000, this.location.x - 30, this.location.y + 35);
        pop();
    }
    if (this.sig){
        push();
            fill(0, 102, 153);
            text("\u03c3= "+Math.round(this.sig * 1000000)/1000000, this.location.x - 30, this.location.y-32);
        pop();
    }
    this.r = lerp(this.r, 32, 0.1);
  }
  
  this.display = function() {
    stroke(0);
    strokeWeight(1);
    push();
        strokeWeight(3);
        fill('rgba(0, 220, 236, .2)');
        ellipse(this.location.x, this.location.y, this.r, this.r);
    pop();
    if (this.backconnects.length != 0) {
        push();
            fill(0, 102, 153);
            text("b= "+Math.round(this.bias * 10000) / 10000, this.location.x - 30, this.location.y + 35);
        pop();
    }
    if (this.sig){
        push();
            fill(0, 102, 153);
            text("\u03c3= "+Math.round(this.sig * 1000000)/1000000, this.location.x - 30, this.location.y-32);
        pop();
    }
    
    // Shrink interpolation
    this.r = lerp(this.r,32,0.1);
  }
}

// ###################
// Connection Object
// ###################

var Connection = function(a, b, weight) {
  // Connection is from Neuron A to B
  this.a = a;
  this.b = b;
  this.weight = weight || random(-1, 1);
  this.sending = false;
  this.output = 0;
  this.sender = createVector(this.a.x, this.a.y);
  this.learningRate = .3;
  this.lastVal;
  this.backing = false;
  this.batch = false;
  //For backprop 
  this.da;
  this.dw;
  
  // The Connection is active
  this.feedforward = function(val) {
    this.lastVal = val;
    this.output = val*this.weight;        // Compute output
    this.sender.x = this.a.location.x;
    this.sender.y = this.a.location.y;
    this.sending = true;             // Turn on sending
    if (this.batch) {
        this.b.feedforward(this.output);
        this.sending = false;
        this.batch = false;

    }
  }

  //Backpropagation of weight
  this.backprop = function(grad) {
    //Accepts gradient coming from prior node
    this.dw = this.lastVal * grad;
    this.da = this.weight * grad;
    this.weight += -this.learningRate * this.dw;
    this.sender.x = this.b.location.x;
    this.sender.y = this.b.location.y;
    this.backing = true;
    if (this.batch) {
        this.a.backprop(this.da);
        this.backing = false;
        this.batch = false;
    }

  }
  
  // Update traveling sender
  this.update = function() {
    if (this.sending) {
      // Use a simple interpolation
      this.sender.x = lerp(this.sender.x, this.b.location.x, 0.1);
      this.sender.y = lerp(this.sender.y, this.b.location.y, 0.1);
      dist = p5.Vector.dist(this.sender, this.b.location);
      // If we've reached the end
      if (dist < 1) {
        // Pass along the output!
        this.b.feedforward(this.output);
        this.sending = false;
      }
    } else if (this.backing) {
        this.sender.x = lerp(this.sender.x, this.a.location.x, 0.1);
        this.sender.y = lerp(this.sender.y, this.a.location.y, 0.1);
        dist = p5.Vector.dist(this.sender, this.a.location);
        if (dist < 1) {
            this.a.backprop(this.da);
            this.backing = false;
        }
    }
  }

  this.backDisplay = function () {
    stroke(0);
    push();
        strokeWeight(3);
        seed = map(this.weight, -1, 1, 0, 1);
        from = color(42, 71, 255);
        to = color(229, 119, 73);
        hue = lerpColor(from, to, seed);
        stroke(hue);
        line(this.a.location.x, this.a.location.y, this.b.location.x, this.b.location.y);
    pop();
    fill(0);

    push();
        angle = p5.Vector.angleBetween(this.a.location, this.b.location);
        strokeWeight(1);
        translate(((this.b.location.x + this.a.location.x)/2.0)
            ,((this.b.location.y + this.a.location.y)/2.0));
        if (this.a.location.y > this.b.location.y) {
            rotate(-angle);
        } else {
            rotate(angle);
        }
        fill(255, 111, 99);
        textSize(18);
        strokeWeight(3);
        text(Math.round(this.weight*10000)/10000,-20,0);
    pop();

    if (this.backing) {
        //moving elipse backwards
        fill(127,47,155);
        strokeWeight(1);
        ellipse(this.sender.x, this.sender.y, 16, 16);
    }
    
  }
  
  // Draw line and traveling circle
  this.display = function() {
    stroke(0);
    push();
        strokeWeight(3);
        seed = map(this.weight, -1, 1, 0, 1);
        from = color(42, 71, 255);
        to = color(229, 119, 73);
        hue = lerpColor(from, to, seed);
        stroke(hue);
        line(this.a.location.x, this.a.location.y, this.b.location.x, this.b.location.y);
    pop();
    fill(0);
    push();
        angle = p5.Vector.angleBetween(this.a.location, this.b.location);
        strokeWeight(1);
        translate(((this.b.location.x + this.a.location.x)/2.0)
            ,((this.b.location.y + this.a.location.y)/2.0));
        if (this.a.location.y > this.b.location.y) {
            rotate(-angle);
        } else {
            rotate(angle);
        }
        fill(37, 219, 90);
        textSize(18);
        strokeWeight(3);
        text(Math.round(this.weight*10000)/10000,-20,0);
    pop();

    if (this.sending) {
      fill(0, 220, 236);
      strokeWeight(1);
      ellipse(this.sender.x, this.sender.y, 16, 16);
    }
  }
}

var Network = function (inBoxes, layers, neurons, connections, location) { 
  
  // The Network has a list of neurons
  this.state = 'forward'
  this.neurons = neurons || [];
  this.connections = connections || [];
  this.location = location || createVector(0, 0);
  this.inBoxes = inBoxes || [];
  this.layers = layers || [];
  this.targetlocs = [];
  this.targets = [0, 1, 0];
  this.outputs = [];

  this.createTargets = function () {

      for (i=this.layers[this.layers.length - 1]; i>0; i--){
        loc = this.neurons[this.neurons.length - 4 + i].location;
        newloc = createVector(loc.x + 120, loc.y);
        this.targetlocs[i - 1] = newloc;
        //Bootstrapping this to keep track of outputs
        this.outputs[i - 1] = this.neurons[this.neurons.length - 4 + i];
      }
    }

  // We can add a Neuron
  this.addNeuron = function (n) {
    this.neurons.push(n);
  }

  // We can connection two Neurons
  this.connect = function (a, b, weight) {
    c = new Connection(a, b, weight);
    a.addConnection(c);
    b.backConnect(c);
    this.connections.push(c);
  } 

  
  // Sending an input to the first Neuron
  // We should do something better to track multiple inputs
  this.feedforward = function() {
    if (this.state == 'batch'){
        console.log("batch forward Network");
        n1 = this.neurons[0];
        n1.batch = true;
        n1.fire();
        
        n2 = this.neurons[1];
        n2.batch = true;
        n2.fire();

        n3 = this.neurons[2];
        n3.batch = true;
        n3.fire();
    } else {
        n1 = this.neurons[0];
        n1.fire();
        
        n2 = this.neurons[1];
        n2.fire();

        n3 = this.neurons[2];
        n3.fire();
    }
    
  }

  //Backward prop
  this.backprop = function() {
    //Get predictions and prep softmax normalization
    outs = [];
    sum = 0;
    for (i in this.outputs){
        outs[i] = this.outputs[i].sig;
        sum += Math.exp(outs[i]);
    }

    //normalize and gradient of loss (softmax)
    douts = [];
    for (i in outs) {
        outs[i] = Math.exp(outs[i]) / sum;
        if (this.targets[i] == 1){
            douts[i] = outs[i] - 1;
        } else {
            douts[i] = outs[i];
        }
    }
    //pass loss values onto neurons for dw,db calcs
    if (this.state == 'batch'){
        for (i in this.outputs) {
            this.outputs[i].batch = true;
            this.outputs[i].backprop(douts[i]);
        }
    } else {
        for (i in this.outputs) {
            this.outputs[i].backprop(douts[i]);
        }
    }
  }
  
  // Update the connections
  this.update = function() {

    for (c in this.connections) {
      this.connections[c].update();
    }
    
  }

  // Draw everything
  this.display = function() {
    if (this.state == 'forward'){
        push();
            translate(this.location.x, this.location.y);
            for (n in this.neurons) {
              this.neurons[n].display();
            }

            for (c in this.connections) {
              this.connections[c].display();
            }
        pop();
    } else if (this.state == 'backward') {
        push();
            translate(this.location.x, this.location.y);
            for (n in this.neurons) {
              this.neurons[n].backDisplay();
            }

            for (c in this.connections) {
              this.connections[c].backDisplay();
            }
        pop();

    }
    push();
        fill(0);
        for (t in this.targets) {
            textSize(32);
            text(this.targets[t], this.targetlocs[t].x + 26, this.targetlocs[t].y);
        }
        textSize(42);
        text("Targets", this.targetlocs[0].x - 40, this.targetlocs[0].y - 50);
    pop();
  }
}

function add(a, b) {
    return a + b;
}

function setup() {
    var myCanvas = createCanvas(windowWidth, windowHeight);
    myCanvas.parent('processing');
    var layers = [3, 2, 3];
    //var numNeurons = layers.reduce(add);
    var neurons = [[], [], []];

    //Make these percentages later
    var forward = createButton('Forward');
    forward.position(windowWidth - 600, windowHeight - 50);
    forward.mousePressed(goForth);
    var backward = createButton('Backprop');
    backward.position(windowWidth - 800, windowHeight - 50);
    backward.mousePressed(backProp);
    var batch = createButton('Batch');
    batchIn = createInput(20);
    batchIn.position(windowWidth - 300, windowHeight - 50);
    batch.position(windowWidth - 400, windowHeight - 50);
    batch.mousePressed(batchTime);


    //Setting up text boxes for the input layers
    var inLayers = [];
    for (i = 0; i<layers[0]; i++){
        rando = Math.round(random(-1,1)*10000)/10000.0;
        input = createInput(rando);
        inLayers[i] = input;
    }

    myNetwork = new Network(inLayers, layers);

    //Create Neurons in order
    var currLayer = 0;
    for (i in layers){
        currLayer ++;
        for (var j = 0; j<layers[i]; j++){
            x = (currLayer * width) / 4;
            y = ((j+1)*height) / (layers[i] + 1);

            if (currLayer == 1) {
                input = inLayers[j];
                input.position(x-100, y-32);
                n = new Neuron(createVector(x,y), random(-1, 1), input.value());
                neurons[currLayer-1].push(n);
                myNetwork.addNeuron(n);

            } else { 
                n = new Neuron(createVector(x,y), random(-1, 1));
                neurons[currLayer-1].push(n);
                myNetwork.addNeuron(n);
            }
        }
    }

    // Create Connections
    for (var l = 0; l < layers.length; l++) {
        for (from in neurons[l]) {
            for (to in neurons[l+1]){
                myNetwork.connect(neurons[l][from], neurons[l+1][to], random(-1,1));
            }
        }
    }

    myNetwork.createTargets();


}

function draw() {
    background(255);
    myNetwork.update();
    myNetwork.display();
}

function goForth() {
    myNetwork.state = 'forward';
    for (i in myNetwork.inBoxes) {
        myNetwork.neurons[i].val = myNetwork.inBoxes[i].value();
    }
    myNetwork.feedforward();
}

function backProp() {
    myNetwork.state = 'backward';
    myNetwork.backprop();
}

function batchTime() {
    noLoop();
    for (i in myNetwork.inBoxes) {
        myNetwork.neurons[i].val = myNetwork.inBoxes[i].value();
    }
    myNetwork.state = 'batch';
    numBatch = parseInt(batchIn.value());
    console.log("starting batch");
    for (i = 0; i < numBatch; i++) {
        myNetwork.feedforward();
        myNetwork.backprop();
        console.log('next batch');
    }
    console.log('ending batch');
    loop();
}

