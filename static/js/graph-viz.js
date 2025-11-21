/**
 * Graph Visualization for Gene-Disease Relations
 * Uses D3.js force-directed graph layout
 */

class GraphVisualizer {
    constructor(containerId, width = 800, height = 600) {
        this.container = d3.select(`#${containerId}`);
        this.width = width;
        this.height = height;
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.selectedNode = null;  // Track selected node for highlighting
        this.nodeElements = null;  // Store node selection for highlighting
        this.linkElements = null;  // Store link selection for highlighting
    }

    initialize() {
        // Clear any existing SVG
        this.container.selectAll('*').remove();

        // Create SVG with zoom capability
        this.svg = this.container
            .append('svg')
            .attr('width', '100%')
            .attr('height', this.height)
            .attr('viewBox', `0 0 ${this.width} ${this.height}`)
            .style('background', '#fafafa')
            .style('border-radius', '10px');

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svg.select('.graph-container').attr('transform', event.transform);
            });

        this.svg.call(zoom);

        // Create container for graph elements
        this.svg.append('g').attr('class', 'graph-container');

        // Add legend
        this.addLegend();
    }

    addLegend() {
        const legend = this.svg.append('g')
            .attr('class', 'legend')
            .attr('transform', 'translate(20, 20)');

        const legendData = [
            { label: 'Disease', color: '#ff6b6b', shape: 'rect' },
            { label: 'Gene', color: '#4ecdc4', shape: 'circle' },
            { label: 'High (≥0.9)', color: '#51cf66', lineStyle: 'solid' },
            { label: 'Medium (0.8-0.9)', color: '#ffa94d', lineStyle: 'dashed' },
            { label: 'Low (<0.8)', color: '#ff6b6b', lineStyle: 'dotted' }
        ];

        legendData.forEach((item, i) => {
            const legendRow = legend.append('g')
                .attr('transform', `translate(0, ${i * 25})`);

            if (item.shape === 'circle') {
                legendRow.append('circle')
                    .attr('cx', 10)
                    .attr('cy', 0)
                    .attr('r', 6)
                    .attr('fill', item.color);
            } else if (item.shape === 'rect') {
                legendRow.append('rect')
                    .attr('x', 4)
                    .attr('y', -6)
                    .attr('width', 12)
                    .attr('height', 12)
                    .attr('fill', item.color);
            } else {
                legendRow.append('line')
                    .attr('x1', 0)
                    .attr('y1', 0)
                    .attr('x2', 20)
                    .attr('y2', 0)
                    .attr('stroke', item.color)
                    .attr('stroke-width', 2)
                    .attr('stroke-dasharray', item.lineStyle === 'dashed' ? '5,5' :
                                            item.lineStyle === 'dotted' ? '2,2' : 'none');
            }

            legendRow.append('text')
                .attr('x', 30)
                .attr('y', 4)
                .style('font-size', '12px')
                .style('fill', '#333')
                .text(item.label);
        });

        // Add background to legend
        const bbox = legend.node().getBBox();
        legend.insert('rect', ':first-child')
            .attr('x', bbox.x - 5)
            .attr('y', bbox.y - 5)
            .attr('width', bbox.width + 10)
            .attr('height', bbox.height + 10)
            .attr('fill', 'white')
            .attr('opacity', 0.9)
            .attr('rx', 5);
    }

    processData(relations) {
        // Extract unique nodes
        const genesSet = new Set();
        const diseasesSet = new Set();

        relations.forEach(rel => {
            genesSet.add(rel.gene);
            diseasesSet.add(rel.disease);
        });

        // Create nodes array
        this.nodes = [];

        // Add disease nodes
        diseasesSet.forEach(disease => {
            this.nodes.push({
                id: disease,
                type: 'disease',
                label: disease,
                group: 'disease'
            });
        });

        // Add gene nodes
        genesSet.forEach(gene => {
            this.nodes.push({
                id: gene,
                type: 'gene',
                label: gene,
                group: 'gene'
            });
        });

        // Create links array
        this.links = relations.map(rel => ({
            source: rel.disease,
            target: rel.gene,
            confidence: rel.confidence,
            evidence: rel.evidence || rel.summary || '',
            relationType: rel.relation_type || 'related',
            value: rel.confidence * 10
        }));

        return { nodes: this.nodes, links: this.links };
    }

    render(relations) {
        if (!relations || relations.length === 0) {
            this.container.html('<div style="text-align:center; padding:50px; color:#666;">No data to visualize</div>');
            return;
        }

        this.initialize();
        const data = this.processData(relations);

        const container = this.svg.select('.graph-container');

        // Create force simulation with spread-out layout
        this.simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links)
                .id(d => d.id)
                .distance(250)  // Increased from 150 - more space between connected nodes
                .strength(0.15))  // Reduced from 0.5 - weaker pull between linked nodes
            .force('charge', d3.forceManyBody()
                .strength(-800)  // Increased repulsion from -300 - pushes nodes apart more
                .distanceMax(800))  // Increased from 400 - repulsion works over longer distance
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(60))  // Increased from 40 - larger collision boundary
            .force('x', d3.forceX(this.width / 2).strength(0.02))  // Gentle pull to center X
            .force('y', d3.forceY(this.height / 2).strength(0.02));  // Gentle pull to center Y

        // Add arrow markers for directed edges
        const defs = this.svg.append('defs');

        // Arrow markers for confidence levels: High (≥0.9), Medium (0.8-0.9), Low (<0.8)
        ['high', 'medium', 'low'].forEach(level => {
            const color = level === 'high' ? '#51cf66' :
                         level === 'medium' ? '#ffa94d' : '#ff6b6b';

            defs.append('marker')
                .attr('id', `arrow-${level}`)
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 25)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('fill', color);
        });

        // Create links
        this.linkElements = container.append('g')
            .selectAll('line')
            .data(data.links)
            .enter()
            .append('line')
            .attr('class', 'link')
            .attr('stroke', d => {
                if (d.confidence >= 0.9) return '#51cf66';  // High: green
                if (d.confidence >= 0.8) return '#ffa94d';  // Medium: yellow
                return '#ff6b6b';  // Low: red
            })
            .attr('stroke-width', d => Math.sqrt(d.value))
            .attr('stroke-dasharray', d => {
                if (d.confidence >= 0.9) return 'none';     // High: solid
                if (d.confidence >= 0.8) return '5,5';      // Medium: dashed
                return '2,2';  // Low: dotted
            })
            .attr('marker-end', d => {
                if (d.confidence >= 0.9) return 'url(#arrow-high)';
                if (d.confidence >= 0.8) return 'url(#arrow-medium)';
                return 'url(#arrow-low)';
            })
            .attr('opacity', 0.6);

        const link = this.linkElements;  // Keep local reference for tick function

        // Create nodes
        this.nodeElements = container.append('g')
            .selectAll('g')
            .data(data.nodes)
            .enter()
            .append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));

        const node = this.nodeElements;  // Keep local reference

        // Add node shapes
        node.append('circle')
            .attr('r', d => d.type === 'disease' ? 0 : 12)
            .attr('fill', d => d.type === 'disease' ? '#ff6b6b' : '#4ecdc4')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer');

        node.append('rect')
            .attr('width', d => d.type === 'disease' ? 20 : 0)
            .attr('height', d => d.type === 'disease' ? 20 : 0)
            .attr('x', -10)
            .attr('y', -10)
            .attr('fill', '#ff6b6b')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('rx', 3)
            .style('cursor', 'pointer');

        // Add labels
        node.append('text')
            .text(d => d.label)
            .attr('x', 0)
            .attr('y', 25)
            .attr('text-anchor', 'middle')
            .style('font-size', '11px')
            .style('font-weight', 'bold')
            .style('fill', '#333')
            .style('pointer-events', 'none');

        // Add tooltips
        const tooltip = d3.select('body').append('div')
            .attr('class', 'graph-tooltip')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('padding', '10px')
            .style('border-radius', '5px')
            .style('font-size', '12px')
            .style('max-width', '300px')
            .style('z-index', '1000');

        node.on('mouseover', function(event, d) {
            tooltip.style('visibility', 'visible')
                .html(`<strong>${d.type.toUpperCase()}</strong><br/>${d.label}<br/><em>Click to highlight connections</em>`);
            d3.select(this).select('circle, rect').attr('stroke-width', 4);
        })
        .on('mousemove', function(event) {
            tooltip.style('top', (event.pageY - 10) + 'px')
                .style('left', (event.pageX + 10) + 'px');
        })
        .on('mouseout', function() {
            tooltip.style('visibility', 'hidden');
            d3.select(this).select('circle, rect').attr('stroke-width', 2);
        })
        .on('click', (event, d) => {
            event.stopPropagation();  // Prevent SVG click from clearing selection
            this.highlightConnections(d);
        });

        // Click on SVG background to clear selection
        this.svg.on('click', () => {
            this.clearHighlight();
        });

        link.on('mouseover', function(event, d) {
            const evidencePreview = d.evidence ? d.evidence.substring(0, 150) + '...' : 'No evidence';
            tooltip.style('visibility', 'visible')
                .html(`<strong>Confidence:</strong> ${(d.confidence * 100).toFixed(0)}%<br/>
                       <strong>Type:</strong> ${d.relationType}<br/>
                       <strong>Evidence:</strong> ${evidencePreview}`);
            d3.select(this).attr('stroke-width', Math.sqrt(d.value) * 2);
        })
        .on('mousemove', function(event) {
            tooltip.style('top', (event.pageY - 10) + 'px')
                .style('left', (event.pageX + 10) + 'px');
        })
        .on('mouseout', function(event, d) {
            tooltip.style('visibility', 'hidden');
            d3.select(this).attr('stroke-width', Math.sqrt(d.value));
        });

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        // Add controls
        this.addControls();
    }

    addControls() {
        const controls = this.container
            .append('div')
            .style('position', 'absolute')
            .style('top', '10px')
            .style('right', '10px')
            .style('background', 'white')
            .style('padding', '10px')
            .style('border-radius', '5px')
            .style('box-shadow', '0 2px 10px rgba(0,0,0,0.1)');

        // Reset button
        controls.append('button')
            .text('Reset View')
            .style('padding', '5px 10px')
            .style('margin', '5px')
            .style('cursor', 'pointer')
            .style('border', '1px solid #667eea')
            .style('background', 'white')
            .style('border-radius', '5px')
            .on('click', () => {
                this.svg.transition().duration(750)
                    .call(d3.zoom().transform, d3.zoomIdentity);
            });

        // Reheat simulation button
        controls.append('button')
            .text('Reorganize')
            .style('padding', '5px 10px')
            .style('margin', '5px')
            .style('cursor', 'pointer')
            .style('border', '1px solid #667eea')
            .style('background', 'white')
            .style('border-radius', '5px')
            .on('click', () => {
                this.simulation.alpha(1).restart();
            });

        // Clear selection button
        controls.append('button')
            .text('Clear Selection')
            .style('padding', '5px 10px')
            .style('margin', '5px')
            .style('cursor', 'pointer')
            .style('border', '1px solid #667eea')
            .style('background', 'white')
            .style('border-radius', '5px')
            .on('click', () => {
                this.clearHighlight();
            });
    }

    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    highlightConnections(selectedNode) {
        this.selectedNode = selectedNode;

        // Find all connected node IDs
        const connectedNodeIds = new Set();
        connectedNodeIds.add(selectedNode.id);

        this.links.forEach(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;

            if (sourceId === selectedNode.id) {
                connectedNodeIds.add(targetId);
            }
            if (targetId === selectedNode.id) {
                connectedNodeIds.add(sourceId);
            }
        });

        // Fade out non-connected nodes
        this.nodeElements
            .transition()
            .duration(300)
            .style('opacity', d => connectedNodeIds.has(d.id) ? 1 : 0.15);

        // Highlight connected nodes with a glow effect
        this.nodeElements
            .select('circle, rect')
            .transition()
            .duration(300)
            .attr('stroke', d => {
                if (d.id === selectedNode.id) return '#667eea';  // Selected node - purple border
                if (connectedNodeIds.has(d.id)) return '#fff';
                return '#fff';
            })
            .attr('stroke-width', d => d.id === selectedNode.id ? 4 : 2);

        // Fade out non-connected links
        this.linkElements
            .transition()
            .duration(300)
            .style('opacity', d => {
                const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                return (sourceId === selectedNode.id || targetId === selectedNode.id) ? 1 : 0.05;
            })
            .attr('stroke-width', d => {
                const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                if (sourceId === selectedNode.id || targetId === selectedNode.id) {
                    return Math.sqrt(d.value) * 1.5;  // Make connected links thicker
                }
                return Math.sqrt(d.value);
            });

        // Update label visibility - show labels only for connected nodes
        this.nodeElements
            .select('text')
            .transition()
            .duration(300)
            .style('opacity', d => connectedNodeIds.has(d.id) ? 1 : 0.1)
            .style('font-weight', d => d.id === selectedNode.id ? 'bold' : 'normal')
            .style('font-size', d => d.id === selectedNode.id ? '13px' : '11px');
    }

    clearHighlight() {
        if (!this.selectedNode) return;

        this.selectedNode = null;

        // Restore all nodes
        this.nodeElements
            .transition()
            .duration(300)
            .style('opacity', 1);

        this.nodeElements
            .select('circle, rect')
            .transition()
            .duration(300)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);

        // Restore all links
        this.linkElements
            .transition()
            .duration(300)
            .style('opacity', 0.6)
            .attr('stroke-width', d => Math.sqrt(d.value));

        // Restore all labels
        this.nodeElements
            .select('text')
            .transition()
            .duration(300)
            .style('opacity', 1)
            .style('font-weight', 'bold')
            .style('font-size', '11px');
    }

    destroy() {
        if (this.simulation) {
            this.simulation.stop();
        }
        this.container.selectAll('*').remove();
        d3.selectAll('.graph-tooltip').remove();
    }
}
