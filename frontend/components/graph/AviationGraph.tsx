"use client";

import { useMemo, useState } from "react";
import type { LiveGraphEdge, LiveGraphNode } from "../../lib/types";

type PositionedNode = LiveGraphNode & { x: number; y: number };

const relationColors: Record<string, string> = {
  AIRCRAFT_ROTATION: "#74c8e3",
  RESOURCE_DEPENDENCY: "#ae9ce9",
  DEPARTS_FROM: "#56717c",
  ARRIVES_AT: "#56717c",
  USES_RESOURCE: "#7b6ea8",
};

function positionNodes(nodes: LiveGraphNode[], edges: LiveGraphEdge[]): PositionedNode[] {
  const flights = nodes.filter((node) => node.type === "flight").sort((a, b) => (a.time ?? 0) - (b.time ?? 0));
  const aircraft = [...new Set(flights.map((node) => node.aircraft ?? "UNASSIGNED"))];
  const times = flights.map((node) => node.time ?? 0);
  const minTime = Math.min(...times, 0);
  const maxTime = Math.max(...times, 1);
  const spread = Math.max(maxTime - minTime, 1);
  const flightPositions = new Map(flights.map((node) => {
    const lane = aircraft.indexOf(node.aircraft ?? "UNASSIGNED");
    return [node.id, {
      ...node,
      x: 90 + (((node.time ?? minTime) - minTime) / spread) * 730,
      y: 160 + lane * Math.max(110, 205 / Math.max(aircraft.length - 1, 1)),
    }];
  }));

  return nodes.map((node, index) => {
    const knownFlight = flightPositions.get(node.id);
    if (knownFlight) return knownFlight;
    const neighbors = edges.flatMap((edge) => {
      if (edge.source === node.id) return [flightPositions.get(edge.target)];
      if (edge.target === node.id) return [flightPositions.get(edge.source)];
      return [];
    }).filter((item): item is PositionedNode => Boolean(item));
    const x = neighbors.length ? neighbors.reduce((sum, neighbor) => sum + neighbor.x, 0) / neighbors.length : 890 + index * 12;
    return { ...node, x, y: node.type === "resource" ? 64 : 405 };
  });
}

function nodeColor(node: LiveGraphNode): string {
  if (node.type === "resource") return "#ae9ce9";
  if (node.type !== "flight") return "#617b85";
  return (node.residual ?? 0) > 8 ? "#e88973" : "#74c8e3";
}

export default function AviationGraph({ nodes, edges, affected }: { nodes: LiveGraphNode[]; edges: LiveGraphEdge[]; affected: string[] }) {
  const graph = useMemo(() => positionNodes(nodes, edges), [nodes, edges]);
  const positions = useMemo(() => new Map(graph.map((node) => [node.id, node])), [graph]);
  const [selected, setSelected] = useState(affected[0] ?? graph.find((node) => node.type === "flight")?.id ?? "");

  return <div className="graph-canvas" aria-label="Observable aviation graph rendered from the public Python bridge">
    <svg className="graph-svg" viewBox="0 0 1000 460" role="img" aria-label="Operational flight, airport, and resource relation graph">
      <title>Observable operational topology returned by the public Python bridge</title>
      <g className="graph-grid" aria-hidden="true">
        <path d="M45 100H955M45 230H955M45 360H955" />
        <path d="M160 28V432M390 28V432M620 28V432M850 28V432" />
      </g>
      <g className="graph-edges">
        {edges.map((edge, index) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) return null;
          const structural = edge.relation === "AIRCRAFT_ROTATION" || edge.relation === "RESOURCE_DEPENDENCY";
          const focused = affected.includes(edge.source) || affected.includes(edge.target);
          return <line key={`${edge.source}-${edge.target}-${edge.relation}-${index}`} x1={source.x} y1={source.y} x2={target.x} y2={target.y}
            className={structural ? "graph-edge structural" : "graph-edge"}
            stroke={relationColors[edge.relation] ?? "#526976"} opacity={structural ? (focused ? 0.95 : 0.64) : 0.28}>
            <title>{`${edge.relation}: ${edge.source} → ${edge.target}`}</title>
          </line>;
        })}
      </g>
      <g className="graph-nodes">
        {graph.map((node) => {
          const flight = node.type === "flight";
          const focused = affected.includes(node.id);
          const isSelected = selected === node.id;
          const color = nodeColor(node);
          return <g key={node.id} className="graph-node" tabIndex={0} role="button" aria-label={`Select ${node.id}`} onClick={() => setSelected(node.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelected(node.id); }}>
            {isSelected && <circle cx={node.x} cy={node.y} r={flight ? 18 : 15} fill={color} opacity="0.14" />}
            <circle cx={node.x} cy={node.y} r={flight ? (focused ? 8 : 6) : 5} fill={color} className={focused ? "affected-node" : ""} />
            {(flight || isSelected) && <text x={node.x} y={node.y - (flight ? 15 : 11)} className="graph-label" textAnchor="middle">{flight ? `${node.id} / ${node.aircraft ?? "—"}` : node.id.replace("resource:", "")}</text>}
            <title>{`${node.id} · ${node.type}${node.aircraft ? ` · ${node.aircraft}` : ""}${node.residual !== undefined ? ` · residual ${node.residual.toFixed(2)}` : ""}`}</title>
          </g>;
        })}
      </g>
    </svg>
    <div className="graph-legend"><span><i className="dot blue" />flight</span><span><i className="dot purple" />resource</span><span><i className="line cyan" />aircraft rotation</span><span><i className="line purple" />resource dependency</span><span><i className="line" />airport association</span></div>
  </div>;
}
