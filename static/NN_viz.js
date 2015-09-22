//Michael Skirpan - Fast Forward Labs
//Visualization of simple Feed-forward and backpropagation algorithm
//Many thanks to:
//Daniel Shiffman - http://natureofcode.com - for education and example code
//P5.js folks - http://p5js.org - for example code
//Michael Nielson - http://neuralnetworksanddeeplearning.com/ - for lucid explanations of algorithms

var myNetwork;

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
    this.learningRate = .01;
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
    console.log('neuron backprop');

    //derivative of sigmoid
    dsig = ((1 - this.sig) * this.sig) * grad;

    //update bias
    this.bias += (-this.learningRate * dsig)

    //shrink for backprop
    this.r = 12;

    //send update for weight
    for (bc in this.backconnects) {
        this.backconnects[bc].backprop(dsig);
    }
  }
  
  this.fire = function() {
    //make it bigger
    this.r = 64;   

    
    // We send the output through all connections
    for (c in this.connections) {
       this.connections[c].feedforward(this.val);
    } 
  }

  this.backDisplay = function() {
    stroke(0);
    strokeWeight(1);
    // Brightness is mapped to sum
    b = map(this.val,0,1,255,0);
    fill(b);
    ellipse(this.location.x, this.location.y, this.r, this.r);
    if (this.backconnects.length != 0) {
        push();
            fill(0, 102, 153);
            text(Math.round(this.bias * 10000) / 10000, this.location.x - 15, this.location.y + 35);
        pop();
    }
    if (this.sig){
        push();
            fill(0, 102, 153);
            text(this.sig.toString(), this.location.x, this.location.y-32);
        pop();
    }
    this.r = lerp(this.r, 32, 0.1);
  }
  
  this.display = function() {
    stroke(0);
    strokeWeight(1);
    // Brightness is mapped to sum
    b = map(this.val,0,1,255,0);
    fill(b);
    ellipse(this.location.x, this.location.y, this.r, this.r);
    if (this.backconnects.length != 0) {
        push();
            fill(0, 102, 153);
            text(Math.round(this.bias * 10000) / 10000, this.location.x - 15, this.location.y + 35);
        pop();
    }
    if (this.sig){
        push();
            fill(0, 102, 153);
            text(this.sig.toString(), this.location.x, this.location.y-32);
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
  this.learningRate = .1;
  this.lastVal;
  this.backing = false;
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
  }

  //Backpropagation of weight
  this.backprop = function(grad) {
    //Accepts gradient coming from prior node
    console.log('connection backprop');
    this.dw = this.lastVal * grad;
    this.da = this.weight * grad;
    this.weight += -this.learningRate * this.dw;
    this.sender.x = this.b.location.x;
    this.sender.y = this.b.location.y;
    this.backing = true;
    console.log(this);
    console.log(this.backing);

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
        console.log('updating connection');
        this.sender.x = lerp(this.sender.x, this.a.location.x, 0.1);
        this.sender.y = lerp(this.sender.y, this.a.location.y, 0.1);
        dist = p5.Vector.dist(this.sender, this.a.location);
        if (dist < 1) {
            console.log('about to backprop connection');
            this.a.backprop(this.da);
            this.backing = false;
        }
    }
  }

  this.backDisplay = function () {
    stroke(0);
    strokeWeight(1+(this.weight*4));
    line(this.a.location.x, this.a.location.y, this.b.location.x, this.b.location.y);
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
        text(Math.round(this.weight*10000)/10000,0,0);
    pop();

    if (this.backing) {
        //moving elipse backwards
        console.log('should be showing ellipse');
        fill(127,47,155);
        strokeWeight(1);
        ellipse(this.sender.x, this.sender.y, 16, 16);
    }
    
  }
  
  // Draw line and traveling circle
  this.display = function() {
    stroke(0);
    strokeWeight(1+(this.weight*4));
    line(this.a.location.x, this.a.location.y, this.b.location.x, this.b.location.y);
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
        fill(0, 198, 229);
        textSize(18);
        text(Math.round(this.weight*10000)/10000,0,0);
    pop();

    if (this.sending) {
      fill(127,47,155);
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
    n1 = this.neurons[0];
    n1.fire();
    
    n2 = this.neurons[1];
    n2.fire();

    n3 = this.neurons[2];
    n3.fire();
    
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
    for (i in this.outputs) {
        this.outputs[i].backprop(douts[i]);
    }
  }
  
  // Update the animation
  this.update = function() {
    // Update backprop, eventually
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
            text(this.targets[t], this.targetlocs[t].x, this.targetlocs[t].y);
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
    var forward = createButton('Forward');
    forward.position(windowWidth - 500, windowHeight - 50);
    forward.mousePressed(goForth);
    var backward = createButton('Backprop');
    backward.position(windowWidth - 800, windowHeight - 50);
    backward.mousePressed(backProp);


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
    newVals = [];
    for (i in myNetwork.inBoxes) {
        myNetwork.neurons[i].val = myNetwork.inBoxes[i].value();
    }
    myNetwork.feedforward();
}

function backProp() {
    myNetwork.state = 'backward'
    myNetwork.backprop();
}

